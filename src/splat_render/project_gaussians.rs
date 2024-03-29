use burn::{
    backend::{
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        wgpu::{
            into_contiguous, AutoGraphicsApi, DynamicKernel, DynamicKernelSource, FloatElement, GraphicsApi, IntElement, JitBackend, JitTensor, SourceTemplate, WgpuRuntime, WorkGroup
        },
        Autodiff,
    },
    tensor::Shape,
};

type BurnBack = JitBackend<WgpuRuntime<AutoGraphicsApi, f32, i32>>;
type BurnBackInner = WgpuRuntime<AutoGraphicsApi, f32, i32>;

use derive_new::new;
use glam::{uvec2, uvec3, UVec3, Vec3};
use crate::camera::Camera;
use super::{gen, AutodiffBackend, Backend, FloatTensor};

#[derive(new, Debug)]
struct ProjectSplats {}

impl DynamicKernelSource for ProjectSplats {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_forward::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl ProjectSplats  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

#[derive(new, Debug)]
struct MapGaussiansToIntersect {
}

impl DynamicKernelSource for MapGaussiansToIntersect {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::map_gaussian_to_intersects::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl MapGaussiansToIntersect  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}


#[derive(new, Debug)]
struct GetTileBinEdges {
}

impl GetTileBinEdges  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}

impl DynamicKernelSource for GetTileBinEdges {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::get_tile_bin_edges::SHADER_STRING)
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(new, Debug)]
struct RasterizeForward {}

impl DynamicKernelSource for RasterizeForward {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl RasterizeForward  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 16, 1)
    }
}


#[derive(new, Debug)]
struct RasterizeBackwards {}

impl DynamicKernelSource for RasterizeBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::rasterize_backwards::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl RasterizeBackwards  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 16, 1)
    }
}


#[derive(new, Debug)]
struct ProjectBackwards {}

impl DynamicKernelSource for ProjectBackwards {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(gen::project_backwards::SHADER_STRING)
    }
    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

impl ProjectBackwards  {
    fn workgroup_size() -> UVec3 {
        uvec3(16, 1, 1)
    }
}


fn get_workgroup(executions: UVec3, group_size: UVec3) -> WorkGroup {
    let execs = (executions.as_vec3() / group_size.as_vec3()).ceil().as_uvec3();
    WorkGroup::new(execs.x, execs.y, execs.z)
}

pub fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for JitBackend<WgpuRuntime<G, F, I>> {
    fn render_gaussians(
        _camera: &Camera,
        _means: FloatTensor<Self, 2>,
        _scales: FloatTensor<Self, 2>,
        _quats: FloatTensor<Self, 2>,
        _colors: FloatTensor<Self, 2>,
        _opacity: FloatTensor<Self, 1>,
        _background: glam::Vec3,
    ) -> FloatTensor<Self, 3> {
        // Implement inference only version. This shouln't be hard but burn makes it a bit annoying!
        todo!();
    }
}

// Create our zero-sized type that will implement the Backward trait.
#[derive(Debug)]
struct RenderBackwards;

impl<C: CheckpointStrategy> Backend for Autodiff<BurnBack, C> {
    fn render_gaussians(
        camera: &Camera,
        means_diff: FloatTensor<Self, 2>,
        scales_diff: FloatTensor<Self, 2>,
        quats_diff: FloatTensor<Self, 2>,
        colors_diff: FloatTensor<Self, 2>,
        opacity_diff: FloatTensor<Self, 1>,
        background: glam::Vec3,
    ) -> FloatTensor<Self, 3> {
        #[derive(Debug, Clone)]
        struct GaussianBackwardState {
            cam: Camera,
            background: Vec3,

            // Splat inputs.
            means: JitTensor<BurnBackInner, f32, 2>,
            scales: JitTensor<BurnBackInner, f32, 2>,
            quats: JitTensor<BurnBackInner, f32, 2>,
            colors: JitTensor<BurnBackInner, f32, 2>,
            opacity: JitTensor<BurnBackInner, f32, 1>,

            // Calculated state.
            radii: JitTensor<BurnBackInner, f32, 1>,
            compensation: JitTensor<BurnBackInner, f32, 1>,
            gaussian_ids_sorted: JitTensor<BurnBackInner, i32, 1>,
            tile_bins: JitTensor<BurnBackInner, i32, 2>,
            xys: JitTensor<BurnBackInner, f32, 2>,
            conics: JitTensor<BurnBackInner, f32, 2>,
            final_index: JitTensor<BurnBackInner, i32, 2>,
            out_img: JitTensor<BurnBackInner, f32, 3>,
        }

        // Implement the backward trait for the given backend B, the node gradient being of rank D
        // with three other gradients to calculate (means, colors, and opacity).
        impl Backward<BurnBack, 3, 5> for RenderBackwards {
            // Our state that we must build during the forward pass to compute the backward pass.
            // (means)
            type State = GaussianBackwardState;

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                // // Get the nodes of each variable.
                let block_size = 16;
                
                let state = ops.state;
                let v_output = grads.consume::<BurnBack, 3>(&ops.node);
                let camera = state.cam;
                let tile_bounds = uvec2(camera.height.div_ceil(block_size), camera.height.div_ceil(block_size));

                assert!(v_output.shape.dims == [camera.height as usize, camera.width as usize, 4]);

                let client = v_output.client.clone();
                let device = v_output.device.clone();

                let intrins = 
                    // TODO: Does this need the tan business?
                    glam::vec4(
                        (camera.width as f32) / (2.0 * camera.fovx.tan()),
                        (camera.height as f32) / (2.0 * camera.fovy.tan()),
                        (camera.width as f32) / 2.0,
                        (camera.height as f32) / 2.0,
                    );
                let viewmat = camera.transform.inverse();

                let create_empty_f32 = |dim|-> JitTensor<BurnBackInner, f32, 2> {
                    let shape = Shape::new(dim);
                    JitTensor::new(
                        client.clone(),
                        device.clone(),
                        shape.clone(),
                        client.create(&vec![0; shape.num_elements() * core::mem::size_of::<f32>()]),
                    )
                };
                let create_empty_f32_1d = |dim|-> JitTensor<BurnBackInner, f32, 1> {
                    let shape = Shape::new(dim);
                    JitTensor::new(
                        client.clone(),
                        device.clone(),
                        shape.clone(),
                        client.create(&vec![0; shape.num_elements() * core::mem::size_of::<f32>()]),
                    )
                };

                let num_points = state.means.shape.dims[0];
                let v_opacity = create_empty_f32_1d([num_points]);

                let v_colors = create_empty_f32([num_points, 4]);
                let v_conic = create_empty_f32([num_points, 4]);
                let v_xy = create_empty_f32([num_points, 2]);

                println!("Calculating gradients");
                println!("Raster backwards");

                let workgroup = get_workgroup(uvec3(camera.height, camera.width, 1), RasterizeBackwards::workgroup_size());

                client.execute(
                    Box::new(DynamicKernel::new(RasterizeBackwards::new(), workgroup)),
                    &[
                        // Read
                        &state.gaussian_ids_sorted.handle,
                        &state.tile_bins.handle,
                        &state.xys.handle,
                        &state.conics.handle,
                        &state.colors.handle,
                        &state.opacity.handle,
                        &state.final_index.handle,
                        &state.out_img.handle,
                        &v_output.handle,

                        &v_opacity.handle,
                        &v_conic.handle,
                        &v_xy.handle,
                        &v_colors.handle,

                        // Uniforms
                        &client.create(bytemuck::bytes_of(&gen::rasterize_backwards::Uniforms::new(
                            [camera.height, camera.width],
                            tile_bounds.into(),
                            state.background.into()
                        ))),
                    ],
                );

                let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), ProjectBackwards::workgroup_size());

                let v_means = create_empty_f32([num_points, 4]);
                let v_scales = create_empty_f32([num_points, 4]);
                let v_quats = create_empty_f32([num_points, 4]);

                println!("Project backwards");

                client.execute(
                    Box::new(DynamicKernel::new(ProjectBackwards::new(), workgroup)),
                    &[
                        // Read
                        &state.means.handle,
                        &state.scales.handle,
                        &state.quats.handle,

                        &state.radii.handle,
                        &state.conics.handle,
                        &state.compensation.handle,
                        &v_xy.handle,
                        &v_conic.handle,
                        &v_means.handle,
                        &v_scales.handle,
                        &v_quats.handle,

                        // Uniforms
                        &client.create(bytemuck::bytes_of(&gen::project_backwards::Uniforms::new(
                            num_points as u32,
                            1.0,
                            viewmat,
                            intrins,
                            [camera.height, camera.width],
                        ))),
                    ],
                );

                println!("Registering gradients");


                // Read from the state we have setup.
                // let means = checkpointer.retrieve_node_output(means_state);

                // Return gradient of means, atm just as itself. So d/dx means == means? errr... ?
                // let grad_means = means;

                // Register gradients for parent nodes.
                let [mean_parent, scales_parent, quats_parent, colors_parent, opacity_parent] = ops.parents;

                if let Some(node) = mean_parent {
                    grads.register::<BurnBack, 2>(node, v_means);
                }

                if let Some(node) = scales_parent {
                    grads.register::<BurnBack, 2>(node, v_scales);
                }

                if let Some(node) = quats_parent {
                    grads.register::<BurnBack, 2>(node, v_quats);
                }

                if let Some(node) = colors_parent {
                    grads.register::<BurnBack, 2>(node, v_colors);
                }

                if let Some(node) = opacity_parent {
                    grads.register::<BurnBack, 1>(node, v_opacity);
                }
            }
        }

        let prep_nodes = RenderBackwards
        .prepare::<C>(
            [
                means_diff.node.clone(), 
                scales_diff.node.clone(), 
                quats_diff.node.clone(), 
                colors_diff.node.clone(), 
                opacity_diff.node.clone()
            ],
            [
                means_diff.graph.clone(), 
                scales_diff.graph.clone(), 
                quats_diff.graph.clone(), 
                colors_diff.graph.clone(), 
                opacity_diff.graph.clone()
            ],
        )
        // Marks the operation as compute bound, meaning it will save its
        // state instead of recomputing itself during checkpointing
        .compute_bound()
        .stateful();

        let (means, scales, quats, colors, opacity) = (means_diff.primitive, scales_diff.primitive, quats_diff.primitive, colors_diff.primitive, opacity_diff.primitive);

        // Preprocess tensors.
        assert!(means.device == scales.device && means.device == quats.device && means.device == colors.device && means.device == opacity.device);
        let (means, scales, quats, colors, opacity) = (
            into_contiguous(means),
            into_contiguous(scales),
            into_contiguous(quats),
            into_contiguous(colors),
            into_contiguous(opacity),
        );

        let num_points = means.shape.dims[0];
        let batches = [means.shape.dims[0], scales.shape.dims[0], quats.shape.dims[0], colors.shape.dims[0], opacity.shape.dims[0]];
        assert!(batches.iter().all(|&x| x == num_points));

        // Atm these have to be dim=4, to be compatible
        // with wgsl alignment.
        assert!(means.shape.dims[1] == 4);
        assert!(scales.shape.dims[1] == 4);

        // 4D quaternion.
        assert!(quats.shape.dims[1] == 4);

        // Divide screen into block
        let block_size = 16;
        let img_size = [camera.width, camera.height];
        let tile_bounds = uvec2(camera.height.div_ceil(block_size), camera.height.div_ceil(block_size));

        let intrins = 
            // TODO: Does this need the tan business?
            glam::vec4(
                (camera.width as f32) / (2.0 * camera.fovx.tan()),
                (camera.height as f32) / (2.0 * camera.fovy.tan()),
                (camera.width as f32) / 2.0,
                (camera.height as f32) / 2.0,
            );
        let viewmat = camera.transform.inverse();

        let client = means.client.clone();
        let device = means.device.clone();

        // TODO: Must be a faster way to create with nulls.
        // TODO: Move all these helpers somewhere else.
        let create_empty_f32 = |dim|-> JitTensor<BurnBackInner, f32, 2> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<f32>()]),
            )
        };
        let create_empty_f32_1d = |dim|-> JitTensor<BurnBackInner, f32, 1> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<f32>()]),
            )
        };

        let create_empty_u32 = |dim| -> JitTensor<BurnBackInner, i32, 2> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<i32>()]),
            )
        };

        let create_empty_u32_1d = |dim| -> JitTensor<BurnBackInner, i32, 1> {
            let shape = Shape::new(dim);
            JitTensor::new(
                client.clone(),
                device.clone(),
                shape.clone(),
                client.create(&vec![0; shape.num_elements() * core::mem::size_of::<i32>()]),
            )
        };

        let to_vec_u32 = |tensor: &JitTensor<BurnBackInner, i32, 2>| -> Vec<u32> {
            let data = client.read(&tensor.handle).read();
            data.into_iter().array_chunks::<4>().map(u32::from_le_bytes).collect()
        };

        let to_vec_u32_1d = |tensor: &JitTensor<BurnBackInner, i32, 1>| -> Vec<u32> {
            let data = client.read(&tensor.handle).read();
            data.into_iter().array_chunks::<4>().map(u32::from_le_bytes).collect()
        };

        let to_vec_f32 = |tensor: &JitTensor<BurnBackInner, f32, 2>| -> Vec<f32> {
            let data = client.read(&tensor.handle).read();
            data.into_iter().array_chunks::<4>().map(f32::from_le_bytes).collect()
        };

        // TODO: We might only need the data on the client for this not the tensor wrapper?
        let covs3d = create_empty_f32([num_points, 6]);
        let xys = create_empty_f32([num_points, 2]);
        let conics = create_empty_f32([num_points, 4]);

        let depths = create_empty_f32_1d([num_points]);
        let radii = create_empty_f32_1d([num_points]);
        let compensation = create_empty_f32_1d([num_points]);
        let num_tiles_hit = create_empty_u32_1d([num_points]);

        let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), ProjectSplats::workgroup_size());
        client.execute(
            Box::new(DynamicKernel::new(ProjectSplats::new(), workgroup)),
            &[
                // Input tensors.
                &means.handle,
                &scales.handle,
                &quats.handle,
                // Output tensors.
                &covs3d.handle,
                &xys.handle,
                &depths.handle,
                &radii.handle,
                &conics.handle,
                &compensation.handle,
                &num_tiles_hit.handle,
                // Aux data.
                &client.create(bytemuck::bytes_of(&gen::project_forward::Uniforms::new(
                    num_points as u32,
                    viewmat,
                    intrins,
                    img_size,
                    tile_bounds.into(),
                    1.0,
                    0.001,
                    block_size,
                ))),
            ],
        );

        // TODO: CPU emulation for now. Investigate if we can do without a cumulative sum?

        // Read num tiles to CPU.
        let num_tiles_hit = to_vec_u32_1d(&num_tiles_hit);
        // Calculate cumulative sum.
        let cum_tiles_hit: Vec<u32> = num_tiles_hit
            .into_iter()
            .scan(0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        let num_intersects = *cum_tiles_hit.last().unwrap() as usize;
        
        // Reupload to GPU.
        let cum_tiles_hit = client.create(bytemuck::cast_slice::<u32, u8>(&cum_tiles_hit));

        // Each intersection maps to a gaussian.
        let gaussian_ids_unsorted = create_empty_u32([num_intersects, 1]);
        let isect_ids_unsorted = create_empty_u32([num_intersects, 1]);

        // Dispatch one thread per point.
        let workgroup = get_workgroup(uvec3(num_points as u32, 1, 1), MapGaussiansToIntersect::workgroup_size());
        client.execute(
            Box::new(DynamicKernel::new(MapGaussiansToIntersect::new(), workgroup)),
            &[
                // Read
                &xys.handle,
                &depths.handle,
                &radii.handle,
                &cum_tiles_hit,
                // Write
                &isect_ids_unsorted.handle,
                &gaussian_ids_unsorted.handle,
                // Uniforms
                &client.create(bytemuck::bytes_of(&gen::map_gaussian_to_intersects::Uniforms::new(
                    num_points as u32,
                    tile_bounds.into(),
                    block_size,
                ))),
            ],
        );

        // TODO: WGSL Radix sort.
        let isect_ids_unsorted = to_vec_u32(&isect_ids_unsorted);

        let gaussian_ids_unsorted = to_vec_u32(&gaussian_ids_unsorted);
        let sorted_indices = argsort(&isect_ids_unsorted);
        
        let isect_ids_sorted: Vec<_> = sorted_indices.iter().copied().map(|x| isect_ids_unsorted[x]).collect();
        let gaussian_ids_sorted: Vec<_> = sorted_indices.iter().copied().map(|x| gaussian_ids_unsorted[x]).collect();

        let workgroup = get_workgroup(uvec3(num_intersects as u32, 1, 1), GetTileBinEdges::workgroup_size());
        let isect_ids_sorted = client.create(bytemuck::cast_slice::<u32, u8>(&isect_ids_sorted));

        let tile_bins = create_empty_u32([(tile_bounds[0] * tile_bounds[1]) as usize, 2]);

        client.execute(
            Box::new(DynamicKernel::new(GetTileBinEdges::new(), workgroup)),
            &[
                // Read
                &isect_ids_sorted,
                // Write
                &tile_bins.handle,
                // Uniforms
                &client.create(bytemuck::bytes_of(&gen::get_tile_bin_edges::Uniforms::new(
                    num_intersects as u32,
                ))),
            ],
        );

        let img_shape = Shape::new([camera.height as usize, camera.width as usize, 4]);
        let out_img = JitTensor::new(
            client.clone(),
            device.clone(),
            img_shape.clone(),
            client.empty(img_shape.num_elements() * core::mem::size_of::<f32>()),
        );

        let final_index_shape = Shape::new([camera.height as usize, camera.width as usize]);
        let final_index = JitTensor::new(
            client.clone(),
            device.clone(),
            final_index_shape.clone(),
            client.empty(final_index_shape.num_elements() * core::mem::size_of::<u32>()),
        );
        let gaussian_ids_sorted = JitTensor::new(
            client.clone(), device.clone(), 
            Shape::new([num_intersects]),
            client.create(bytemuck::cast_slice::<u32, u8>(&gaussian_ids_sorted))
        );

        let workgroup = get_workgroup(uvec3(img_shape.dims[0] as u32, img_shape.dims[1] as u32, 1), RasterizeForward::workgroup_size());
        client.execute(
            Box::new(DynamicKernel::new(RasterizeForward::new(), workgroup)),
            &[
                // Read
                &gaussian_ids_sorted.handle,
                &tile_bins.handle,
                &xys.handle,
                &conics.handle,
                &colors.handle,
                &opacity.handle,

                // Write
                &out_img.handle,
                &final_index.handle,

                // Uniforms
                &client.create(bytemuck::bytes_of(&gen::rasterize::Uniforms::new(
                    tile_bounds.into(),
                    background.into(),
                    img_size
                ))),
            ],
        );

        // Prepare a stateful operation with each variable node and corresponding graph.
        //
        // Each node can be fetched with `ops.parents` in the same order as defined here.
        match prep_nodes {
            OpsKind::Tracked(mut prep) => {
                let state = GaussianBackwardState {
                    // TODO: Respect checkpointing in this.
                    means,
                    scales,
                    quats,
                    radii,
                    compensation,
                    cam: camera.clone(),
                    background,
                    out_img: out_img.clone(),
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    conics,
                    colors,
                    opacity,
                    final_index,
                };
        
                prep.finish(state, out_img)
            }
            OpsKind::UnTracked(prep) => {
                // When no node is tracked, we can just compute the original operation without
                // keeping any state.
                prep.finish(out_img)
            }
        }
    }
}

impl AutodiffBackend for Autodiff<BurnBack>
{
}