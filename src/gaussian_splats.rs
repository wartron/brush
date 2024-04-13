use std::io::BufReader;

use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    tensor::{activation::sigmoid, Data, Device, Shape},
};
use ndarray::Axis;
use ply_rs::parser::Parser;
use rerun::{Color, RecordingStream};

use crate::{
    camera::Camera,
    splat_render::{self, Backend, RenderArgs},
    utils,
};
use burn::tensor::Distribution;
use burn::tensor::Tensor;

use anyhow::Result;

#[derive(Config)]
pub(crate) struct SplatsConfig {
    num_points: usize,
    aabb_scale: f32,
    max_sh_degree: u32,
    position_lr_scale: f32,
}

impl SplatsConfig {
    pub(crate) fn build<B: Backend>(&self, device: &Device<B>) -> Splats<B> {
        Splats::new(self.num_points, self.aabb_scale, device)
    }
}

// A Gaussian splat model.
// This implementation wraps CUDA kernels from (Kerbel and Kopanas et al, 2023).
#[derive(Module, Debug)]
pub(crate) struct Splats<B: Backend> {
    // f32[n, 3]. Position.
    means: Param<Tensor<B, 2>>,

    // f32[n, sh]. SH coefficients for diffuse color.
    colors: Param<Tensor<B, 2>>,

    // f32[n, 4]. Rotation as quaternion matrices.
    rotation: Param<Tensor<B, 2>>,

    // f32[n]. Opacity parameters.
    opacity: Param<Tensor<B, 1>>,

    // f32[n, 3]. Scale matrix coefficients.
    scales: Param<Tensor<B, 2>>,
}

struct SplatTrainState<B: Backend> {
    // Non trainable params.
    // f32[n]. Maximum projected radius of each Gaussian in pixel-units. It is
    // later used during culling.
    max_radii_2d: Tensor<B, 1>,

    // Helper tensors for accumulating the viewspace_xyz gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    //
    // Sum of gradient norms for each Gaussian in pixel-units. This accumulator
    // is incremented when a Gaussian is visible in a training batch.
    xyz_gradient_accum: Tensor<B, 1>,

    // Number of times a Gaussian is visible in a training batch.
    denom: Tensor<B, 1>,
}

impl<B: Backend> SplatTrainState<B> {
    fn new(num_points: usize, device: &Device<B>) -> Self {
        Self {
            max_radii_2d: Tensor::zeros([num_points], device),
            xyz_gradient_accum: Tensor::zeros([num_points], device),
            denom: Tensor::zeros([num_points], device),
        }
    }
}

fn vec3_to_tensor4<B: Backend>(data: Vec<glam::Vec3>, device: &Device<B>) -> Tensor<B, 2> {
    let mean_vec = data
        .iter()
        .flat_map(|v| [v.x, v.y, v.z, 0.0])
        .collect::<Vec<_>>();
    Tensor::from_data(
        Data::new(mean_vec, Shape::new([data.len(), 4])).convert(),
        device,
    )
}

fn vec4_to_tensor4<B: Backend>(data: Vec<glam::Vec4>, device: &Device<B>) -> Tensor<B, 2> {
    let mean_vec = data
        .iter()
        .flat_map(|v| [v.x, v.y, v.z, v.w])
        .collect::<Vec<_>>();
    Tensor::from_data(
        Data::new(mean_vec, Shape::new([data.len(), 4])).convert(),
        device,
    )
}

fn float_to_tensor<B: Backend>(data: Vec<f32>, device: &Device<B>) -> Tensor<B, 1> {
    Tensor::from_data(
        Data::new(data.clone(), Shape::new([data.len()])).convert(),
        device,
    )
}

impl<B: Backend> Splats<B> {
    pub(crate) fn from_data(data: Vec<GaussianData>, device: &Device<B>) -> Self {
        // TODO: This is all terribly space inefficient.
        let means = data
            .iter()
            .flat_map(|d| [d.means.x, d.means.y, d.means.z, 0.0])
            .collect::<Vec<_>>();
        let colors = data
            .iter()
            .flat_map(|d| [d.colors.x, d.colors.y, d.colors.z, 0.0])
            .collect::<Vec<_>>();
        let rotation = data
            .iter()
            .flat_map(|d| [d.rotation.x, d.rotation.y, d.rotation.z, d.rotation.w])
            .collect::<Vec<_>>();
        let opacity = data.iter().map(|d| d.opacity).collect::<Vec<_>>();
        let scales = data
            .iter()
            .flat_map(|d| [d.scale.x, d.scale.y, d.scale.z, 0.0])
            .collect::<Vec<_>>();

        let num_points = data.len();

        Splats {
            means: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    Data::new(means, Shape::new([num_points, 4])).convert(),
                    device,
                )
                .require_grad(),
            ),
            colors: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    Data::new(colors, Shape::new([num_points, 4])).convert(),
                    device,
                )
                .require_grad(),
            ),
            rotation: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    Data::new(rotation, Shape::new([num_points, 4])).convert(),
                    device,
                )
                .require_grad(),
            ),
            opacity: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    Data::new(opacity, Shape::new([num_points])).convert(),
                    device,
                )
                .require_grad(),
            ),
            scales: Param::initialized(
                ParamId::new(),
                Tensor::from_data(
                    Data::new(scales, Shape::new([num_points, 4])).convert(),
                    device,
                )
                .require_grad(),
            ),
        }
    }

    pub(crate) fn new(num_points: usize, aabb_scale: f32, device: &Device<B>) -> Splats<B> {
        let extent = (aabb_scale as f64) / 2.0;
        let means = Tensor::random(
            [num_points, 4],
            Distribution::Uniform(-extent, extent),
            device,
        );

        let colors = Tensor::random([num_points, 4], Distribution::Uniform(-2.0, 2.0), device);
        let init_rotation = Tensor::from_floats([1.0, 0.0, 0.0, 0.0], device)
            .unsqueeze::<2>()
            .repeat(0, num_points);

        let init_opacity = Tensor::random([num_points], Distribution::Uniform(-4.0, -2.0), device);

        // TODO: Fancy KNN init.
        let init_scale = Tensor::random([num_points, 4], Distribution::Uniform(-5.0, -3.0), device);

        // TODO: Support lazy loading.
        // Model parameters.
        Splats {
            means: Param::initialized(ParamId::new(), means.require_grad()),
            colors: Param::initialized(ParamId::new(), colors.require_grad()),
            rotation: Param::initialized(ParamId::new(), init_rotation.require_grad()),
            opacity: Param::initialized(ParamId::new(), init_opacity.require_grad()),
            scales: Param::initialized(ParamId::new(), init_scale.require_grad()),
        }
    }

    // Args:
    //   cfg: ...
    //   position_lr_scale: Multiplier for learning rate for positions.  Larger
    //     values mean higher learning rates.

    // Updates rolling statistics that we capture during rendering.
    // pub(crate) fn update_rolling_statistics(&mut self, render_pkg: RenderPackage<B>) {
    //     let radii = render_pkg.radii;
    //     let visible_mask = radii.clone().greater_elem(0.0);
    //     // TODO: This is not as efficient as could be...
    //     // Want these operations to be sparse.
    //     // TODO: Use max_pair.
    //     self.max_radii_2d = radii.clone().mask_where(
    //         visible_mask.clone(),
    //         Tensor::cat(
    //             vec![radii.unsqueeze(), self.max_radii_2d.clone().unsqueeze()],
    //             0,
    //         )
    //         .max_dim(0),
    //     );
    //     // TODO: How do we get grads here? Would need to be sure B: AutoDiffBackend.
    //     // let grad = screenspace_points.
    //     // self.xyz_gradient_accum[visibility_filter] += torch.norm(
    //     //     screenspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True
    //     // );
    //     self.denom = self.denom.clone() + visible_mask.float();
    // }

    /// Resets all the opacities to 0.01.
    pub(crate) fn reset_opacity(&mut self) {
        // self.opacity =
        //     utils::inverse_sigmoid(Tensor::zeros_like(&self.opacity.val()) + 0.01).into();
        // TODO: Wtf.
        // Update optimizer with the new tensor
        //   optimizable_tensors = gs_adam_helpers.replace_tensor_to_optimizer(
        //       self.optimizer, opacities_new, 'opacity'
        //   );
        //   // Make sure that the tensor we are storing is the same tensor the
        //   // optimizer is optimizing
        //   self.opacity = optimizable_tensors['opacity'];
    }

    // // Densifies and prunes the Gaussians.
    // // Args:
    // //   max_grad: See densify_by_clone() and densify_by_split().
    // //   min_opacity_threshold: Gaussians with an opacity lower than this will be
    // //     deleted.
    // //   max_pixel_threshold: Optional. If specified, prune Gaussians whose radius
    // //     is larger than this in pixel-units.
    // //   max_world_size_threshold: Optional. If specified, prune Gaussians whose
    // //     radius is larger than this in world coordinates.
    // //   clone_vs_split_size_threshold: See densify_by_clone() and
    // //     densify_by_split().
    // fn densify_and_prune(
    //     self,
    //     max_grad: f32,
    //     min_opacity_threshold: f32,
    //     max_pixel_threshold: f32,
    //     max_world_size_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     device: &Device<B>,
    // ) {

    //   // f32[n,1]. Compute average magnitude of the gradient for each Gaussian in
    //   // pixel-units while accounting for the number of times each Gaussian was
    //   // seen during training.
    //   let grads = self.xyz_gradient_accum / self.denom;
    //   grads[grads.isnan()] = 0.0;

    //   self.densify_by_clone(grads, max_grad, clone_vs_split_size_threshold, device);
    //   self.densify_by_split(grads, max_grad, clone_vs_split_size_threshold, 2, device);

    //   // bool[n]. If True, delete these Gaussians.
    //   let prune_mask = (
    //       self.opacity_activation(self.opacity) < min_opacity_threshold
    //   ).squeeze();

    //   if let Some(threshold) = max_pixel_threshold {
    //     // Delete Gaussians with too large of a radius in pixel-units.
    //     let big_points_vs = self.max_radii_2d > max_pixel_threshold;

    //     // Delete Gaussians with too large of a radius in world-units.
    //     let big_points_ws =
    //         self.scale_activation(self.scale).max_dim(1).values
    //         > max_world_size_threshold;

    //     let prune_mask = Tensor::logical_or(
    //         Tensor::logical_or(prune_mask, big_points_vs), big_points_ws
    //     );
    // }

    //   self.prune_points(prune_mask);
    // }

    // // Prunes points based on the given mask.
    // //
    // // Args:
    // //   mask: bool[n]. If True, prune this Gaussian.
    // fn prune_points(&mut self, mask: Tensor<B, 2>) {
    //     // TODO: Ehh not sure how/what.
    // //   let valid_points_mask = 1.0 - mask;

    // //   let optimizable_tensors = gs_adam_helpers.prune_optimizer(
    // //       self.optimizer, valid_points_mask
    // //   );

    // //   self.xyz = optimizable_tensors['xyz'];
    // //   self.sh_dc = optimizable_tensors['sh_dc'];
    // //   self.sh_rest = optimizable_tensors['sh_rest'];
    // //   self.opacity = optimizable_tensors['opacity'];
    // //   self.scale = optimizable_tensors['scale'];
    // //   self.rotation = optimizable_tensors['rotation'];

    // //   self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask];
    // //   self.denom = self.denom[valid_points_mask];
    // //   self.max_radii_2d = self.max_radii_2d[valid_points_mask];
    // }

    // // Densifies Gaussians by splitting.

    // // Args:
    // //   grads: f32[n,1]. Average squared magnitude of the gradient for each
    // //     Gaussian in pixel-units.
    // //   grad_threshold: Minimum gradient magnitude for
    // //   clone_vs_split_size_threshold: Threshold on scale in world units.
    // //     Gaussians which meet the gradient condition and have a scale larger than
    // //     this are split into `n_splits` new Gaussians.
    // //   n_splits: Number of new Gaussians to create for each split Gaussian.
    // fn densify_by_split(
    //     &mut self,
    //     grads: Tensor<B, 2>,
    //     grad_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     n_splits: i32,
    //     device: &Device<B>
    // ) {

    //   let n_init_points = self.xyz.dims()[0];
    //   // f32[n]. Extract points that satisfy the gradient condition.
    //   let padded_grad = Tensor::zeros([n_init_points], device);
    //   padded_grad.slice_assign([0..grads.dims()[0]], grads);

    //   // Decide which Gaussians are eligible for splitting or cloning based on
    //   // their gradient magnitude.
    //   let selected_pts_mask = padded_grad >= grad_threshold;

    //   // Gaussians are split if their radius in world-units exceeds a threshold.
    //   selected_pts_mask = Tensor::logical_and(
    //       selected_pts_mask,
    //       Tensor::max_dim(self.scale_activation(self.scale), 1).values
    //       > clone_vs_split_size_threshold,
    //   );

    //   // Sample position of each new Gaussian.
    //   let stds = self.scale_activation(self.scale[selected_pts_mask]).repeat(
    //       n_splits, 1
    //   );
    //   let means = torch.zeros((stds.size(0), 3), device);
    //   let samples = torch.normal(mean=means, std=stds);
    //   let rots = gs_utils.qvec2rotmat(self.rotation[selected_pts_mask]).repeat(
    //       n_splits, 1, 1
    //   );
    //   let new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[
    //       selected_pts_mask
    //   ].repeat(n_splits, 1);

    //   // Set the scale of each new Gaussian to approximately 1/k of its parent.
    //   let new_scale = self.inverse_scale_activation(
    //       self.scale_activation(self.scale[selected_pts_mask]).repeat(n_splits, 1)
    //       / (0.8 * n_splits)
    //   );

    //   // Split Gaussians inherit remaining properties from their parent.
    //   let new_rotation = self.rotation[selected_pts_mask].repeat(n_splits, 1);
    //   let new_sh_dc = self.sh_dc[selected_pts_mask].repeat(n_splits, 1, 1);
    //   let new_sh_rest = self.sh_rest[selected_pts_mask].repeat(n_splits, 1, 1);
    //   let new_opacity = self.opacity[selected_pts_mask].repeat(n_splits, 1);

    //   self.densification_postfix(
    //       new_xyz,
    //       new_sh_dc,
    //       new_sh_rest,
    //       new_opacity,
    //       new_scale,
    //       new_rotation,
    //   );

    //   let prune_filter = torch.cat((
    //       selected_pts_mask,
    //       torch.zeros(
    //           n_splits * selected_pts_mask.sum()
    //       ),
    //   ));

    //   self.prune_points(prune_filter);
    // }

    // // Densifies Gaussians by cloning.
    // //
    // // Args:
    // //   grads: f32[n,1]. Average squared magnitude of the gradient for each
    // //     Gaussian in pixel-units.
    // //   grad_threshold: Minimum gradient magnitude for
    // //   clone_vs_split_size_threshold: Threshold on scale in world units.
    // //     Gaussians which meet the gradient condition and have a scale smaller
    // //     than this are cloned with the exact same parameters.
    // fn densify_by_clone(
    //     &mut self,
    //     grads: Tensor<B, 2>,
    //     grad_threshold: f32,
    //     clone_vs_split_size_threshold: f32,
    //     device: &Device<B>,
    // ) {

    //   // Extract points that satisfy the gradient condition
    //   let selected_pts_mask = Tensor::where(
    //       torch.norm(grads, dim=-1) >= grad_threshold, true, false
    //   );

    //   // From those choose only the ones that are small enough to be cloned
    //   selected_pts_mask = Tensor::logical_and(
    //       selected_pts_mask,
    //       Tensor::max_dim(self.scale_activation(self.scale), 1).values
    //       <= clone_vs_split_size_threshold,
    //   );

    //   let new_xyz = self.xyz[selected_pts_mask];
    //   let new_sh_dc = self.sh_dc[selected_pts_mask];
    //   let new_sh_rest = self.sh_rest[selected_pts_mask];
    //   let new_opacities = self.opacity[selected_pts_mask];
    //   let new_scale = self.scale[selected_pts_mask];
    //   let new_rotation = self.rotation[selected_pts_mask];

    //   self.densification_postfix(
    //       new_xyz,
    //       new_sh_dc,
    //       new_sh_rest,
    //       new_opacities,
    //       new_scale,
    //       new_rotation,
    //   );
    // }

    // // Updates the optimizer by appending the new tensors.
    // fn densification_postfix(
    //     self,
    //     new_xyz: Tensor<B, 2>,
    //     new_features_dc: Tensor<B, 3>,
    //     new_features_rest: Tensor<B, 3>,
    //     new_opacities: Tensor<B, 2>,
    //     new_scale: Tensor<B, 2>,
    //     new_rotation: Tensor<B, 2>,
    // ) {
    //   tensors_dict = {
    //       'xyz': new_xyz,
    //       'sh_dc': new_features_dc,
    //       'sh_rest': new_features_rest,
    //       'opacity': new_opacities,
    //       'scale': new_scale,
    //       'rotation': new_rotation,
    //   };

    //   optimizable_tensors = gs_adam_helpers.cat_tensors_to_optimizer(
    //       self.optimizer, tensors_dict
    //   );

    //   self.xyz = optimizable_tensors['xyz'];
    //   self.sh_dc = optimizable_tensors['sh_dc'];
    //   self.sh_rest = optimizable_tensors['sh_rest'];
    //   self.opacity = optimizable_tensors['opacity'];
    //   self.scale = optimizable_tensors['scale'];
    //   self.rotation = optimizable_tensors['rotation'];

    //   self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device='cuda');
    //   self.denom = torch.zeros((self.xyz.shape[0], 1), device='cuda');
    //   self.max_radii_2d = torch.zeros((self.xyz.shape[0]), device='cuda');
    // }

    // Renders an image by splatting the gaussians.
    // Args:
    //   camera: Camera to render.
    //   bg_color: Background color.
    // Returns:
    //   A tuple of which the first element is the rendered image and the second
    //   elements is a dictionary consisting of statistics that we need to keep
    //   track
    //   during training. More specifically:
    //   * screenspace_points: a tensor that "holds" the viewspace positional
    //     gradients
    //   * visibility_filter: a boolean tensor that indicates which gaussians
    //     participated in the rendering.
    //   * radii: the maximum screenspace radius of each gaussian
    pub(crate) fn render(
        &self,
        camera: &Camera,
        bg_color: glam::Vec3,
    ) -> (
        Tensor<B, 3>,
        crate::splat_render::Aux<crate::splat_render::BurnBack>,
    ) {
        let cur_rot = self.rotation.val();

        let norms = Tensor::sum_dim(cur_rot.clone() * cur_rot.clone(), 1).sqrt();
        let norm_rotation = cur_rot / Tensor::clamp_min(norms, 1e-6);

        let args = RenderArgs { sync_kernels: true };

        splat_render::render::render(
            camera,
            self.means.val(),
            self.scales.val().exp(),
            norm_rotation,
            self.colors.val(),
            sigmoid(self.opacity.val()),
            bg_color,
            args,
        )
    }

    pub(crate) fn visualize(&self, rec: &RecordingStream) -> Result<()> {
        let means_data = utils::burn_to_ndarray(self.means.val());
        let means = means_data
            .axis_iter(Axis(0))
            .map(|c| glam::vec3(c[0], c[1], c[2]));

        let colors_data = utils::burn_to_ndarray(self.colors.val());

        let colors = colors_data.axis_iter(Axis(0)).map(|c| {
            Color::from_rgb(
                (c[[0]] * 255.0) as u8,
                (c[[1]] * 255.0) as u8,
                (c[[2]] * 255.0) as u8,
            )
        });

        let scales_data = utils::burn_to_ndarray(self.scales.val().exp());
        let radii = scales_data
            .axis_iter(Axis(0))
            .map(|c| 0.5 * 0.33 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());

        rec.log(
            "world/splat/points",
            &rerun::Points3D::new(means)
                .with_colors(colors)
                .with_radii(radii),
        )?;
        Ok(())
    }

    pub(crate) fn cur_num_points(&self) -> usize {
        self.means.dims()[0]
    }
}

use ply_rs::ply::{Property, PropertyAccess};

#[derive(Default)]
pub(crate) struct GaussianData {
    means: glam::Vec3,
    colors: glam::Vec3,
    scale: glam::Vec3,
    opacity: f32,
    rotation: glam::Vec4,
}

fn inv_sigmoid(v: f32) -> f32 {
    (v / (1.0 - v)).ln()
}

const SH_C0: f32 = 0.28209479;

impl PropertyAccess for GaussianData {
    fn new() -> Self {
        GaussianData::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            // TODO: SH.
            ("x", Property::Float(v)) => self.means[0] = v,
            ("y", Property::Float(v)) => self.means[1] = v,
            ("z", Property::Float(v)) => self.means[2] = v,

            // TODO: This 0.5 shouldn't be needed anymore once we do full SH?
            ("f_dc_0", Property::Float(v)) => self.colors[0] = v * SH_C0 + 0.5,
            ("f_dc_1", Property::Float(v)) => self.colors[1] = v * SH_C0 + 0.5,
            ("f_dc_2", Property::Float(v)) => self.colors[2] = v * SH_C0 + 0.5,

            ("scale_0", Property::Float(v)) => self.scale[0] = v,
            ("scale_1", Property::Float(v)) => self.scale[1] = v,
            ("scale_2", Property::Float(v)) => self.scale[2] = v,

            ("opacity", Property::Float(v)) => self.opacity = v,

            ("rot_0", Property::Float(v)) => self.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation[3] = v,

            (_, _) => {}
        }
    }
}

pub fn create_from_ply<B: Backend>(file: &str, device: &B::Device) -> Result<Splats<B>> {
    // set up a reader, in this case a file.
    let f = std::fs::File::open(file).unwrap();
    let mut reader = BufReader::new(f);

    let gaussian_parser = Parser::<GaussianData>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    for (_ignore_key, element) in &header.elements {
        if element.name == "vertex" {
            cloud = gaussian_parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    // Return normalized rotations.
    for gaussian in &mut cloud {
        // TODO: Clamp maximum variance? Is that needed?
        // TODO: Is scale in log(scale) or scale format?
        //
        // for i in 0..3 {
        //     gaussian.scale_opacity.scale[i] = gaussian.scale_opacity.scale[i]
        //         .max(mean_scale - MAX_SIZE_VARIANCE)
        //         .min(mean_scale + MAX_SIZE_VARIANCE)
        //         .exp();
        // }
        gaussian.rotation = gaussian.rotation.normalize();
    }
    Ok(Splats::from_data(cloud, device))
}
