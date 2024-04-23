#import helpers;

struct Uniforms {
    // View matrix transform world to view position.
    viewmat: mat4x4f,
    // Focal of camera (fx, fy)
    focal: vec2f,
    // Img resolution (w, h)
    img_size: vec2u,
}
@group(0) @binding(0) var<storage> uniforms: Uniforms;

@group(0) @binding(1) var<storage> means: array<vec4f>;
@group(0) @binding(2) var<storage> scales: array<vec4f>;
@group(0) @binding(3) var<storage> quats: array<vec4f>;

@group(0) @binding(4) var<storage> radii: array<i32>;
@group(0) @binding(5) var<storage> cov2ds: array<vec4f>;
@group(0) @binding(6) var<storage> v_xy: array<vec2f>;
@group(0) @binding(7) var<storage> v_conic: array<vec4f>;
@group(0) @binding(8) var<storage> v_opacity: array<f32>;

@group(0) @binding(9) var<storage, read_write> v_means: array<vec4f>;
@group(0) @binding(10) var<storage, read_write> v_scales: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> v_quats: array<vec4f>;

// TODO: presumably this is only relevant when supervising depths?
// @group(0) @binding(11) var<storage> v_depth: array<f32>;

fn project_pix_vjp(fxfy: vec2f, p_view: vec3f, v_xy: vec2f) -> vec3f {
    let rw = 1.0f / (p_view.z + 1e-6f);
    let v_proj = vec2f(fxfy.x * v_xy.x, fxfy.y * v_xy.y);
    return vec3f(v_proj.x * rw, v_proj.y * rw, -(v_proj.x * p_view.x + v_proj.y * p_view.y) * rw * rw);
}

fn quat_to_rotmat_vjp(quat: vec4f, v_R: mat3x3f) -> vec4f {
    let quat_norm = normalize(quat);

    let w = quat_norm.x;
    let x = quat_norm.y;
    let y = quat_norm.z;
    let z = quat_norm.w;

    return vec4f(
        // w element stored in x field
        2.f * (
                  // v_quat.w = 2.f * (
                  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                  z * (v_R[0][1] - v_R[1][0])
              ),
        // x element in y field
        2.f *
        (
            // v_quat.x = 2.f * (
            -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
            z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
        ),
        // y element in z field
        2.f *
        (
            // v_quat.y = 2.f * (
            x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
            z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
        ),
        // z element in w field
        2.f *
        (
            // v_quat.z = 2.f * (
            x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
            2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
        )
    );
}

fn cov2d_to_conic_vjp(conic: vec3f, v_conic: vec3f) -> vec3f {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    let X = mat2x2f(conic.x, conic.y, conic.y, conic.z);
    let G = mat2x2f(v_conic.x, v_conic.y / 2.0, v_conic.y / 2.0, v_conic.z);
    let v_Sigma = X * G * X;

    return -vec3f(
        v_Sigma[0][0],
        v_Sigma[1][0] + v_Sigma[0][1],
        v_Sigma[1][1]
    );
}

@compute
@workgroup_size(helpers::SPLATS_PER_GROUP, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let idx = global_id.x;

    let num_points = arrayLength(&means);

    if idx >= num_points || radii[idx] == 0 {
        return;
    }

    v_quats[idx] = vec4f(0.0);
    v_scales[idx] = vec4f(0.0);
    v_means[idx] = vec4f(0.0);

    if radii[idx] == 0 {
        return;
    }

    let viewmat = uniforms.viewmat;
    let focal = uniforms.focal;

    let mean = means[idx].xyz;
    let scale = scales[idx].xyz;
    let quat = quats[idx];

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let p_view = W * mean + viewmat[3].xyz;
    var v_mean = transpose(W) * project_pix_vjp(focal, p_view, v_xy[idx]);

    // get v_mean3d from v_xy
    // get z gradient contribution to mean3d gradient
    // TODO: Is this indeed only active if depth is supervised?
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    // let v_z = v_depth[idx];
    // v_mean += viewmat[2].xyz * v_z;

    // get v_cov2d
    // compute vjp from df/d_conic to df/c_cov2d
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    let cov2d = cov2ds[idx].xyz;
    let compensation = helpers::cov_compensation(cov2d);
    let conic = helpers::cov2d_to_conic(cov2d);
    let v_conic = v_conic[idx].xyz;
    var v_cov2d = cov2d_to_conic_vjp(conic, v_conic);
    
    // Compensation is applied as opac * comp
    // so deriv is v_opac.
    // TODO: Re-enable compensation.
    let v_compensation = v_opacity[idx] * 0.0;

    // comp = sqrt(det(cov2d - 0.3 I) / det(cov2d))
    // conic = inverse(cov2d)
    // df / d_cov2d = df / d comp * 0.5 / comp * [ d comp^2 / d cov2d ]
    // d comp^2 / d cov2d = (1 - comp^2) * conic - 0.3 I * det(conic)
    let inv_det = conic.x * conic.z - conic.y * conic.y;
    let one_minus_sqr_comp = 1.0 - compensation * compensation;
    let v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    v_cov2d += vec3f(
        v_sqr_comp * (one_minus_sqr_comp * conic.x - helpers::COV_BLUR * inv_det),
        2.0 * v_sqr_comp * (one_minus_sqr_comp * conic.y),
        v_sqr_comp * (one_minus_sqr_comp * conic.z - helpers::COV_BLUR * inv_det)
    );

    // get v_cov3d (and v_mean3d contribution)
    let rz = 1.0 / p_view.z;
    let rz2 = rz * rz;

    let J = mat3x3f(
        vec3f(focal.x * rz, 0.0f, 0.0f),
        vec3f(0.0f, focal.y * rz, 0.0f),
        vec3f(-focal.x * p_view.x * rz2, -focal.y * p_view.y * rz2, 0.0f)
    );

    let R = helpers::quat_to_rotmat(quat);
    let S = helpers::scale_to_mat(scale);
    let M = R * S;
    let V = M * transpose(M);

    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    let v_cov = mat3x3f(
        vec3f(v_cov2d.x, 0.5 * v_cov2d.y, 0.0),
        vec3f(0.5 * v_cov2d.y, v_cov2d.z, 0.0),
        vec3f(0.0, 0.0, 0.0),
    );

    let T = J * W;
    let Tt = transpose(T);
    let Vt = transpose(V);

    let v_V = Tt * v_cov * T;
    let v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    let v_cov3d0 = v_V[0][0];
    let v_cov3d1 = v_V[0][1] + v_V[1][0];
    let v_cov3d2 = v_V[0][2] + v_V[2][0];
    let v_cov3d3 = v_V[1][1];
    let v_cov3d4 = v_V[1][2] + v_V[2][1];
    let v_cov3d5 = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    let v_J = v_T * transpose(W);
    let rz3 = rz2 * rz;
    let v_t = vec3f(
        -focal.x * rz2 * v_J[2][0],
        -focal.y * rz2 * v_J[2][1],
        -focal.x * rz2 * v_J[0][0] + 2.0 * focal.x * p_view.x * rz3 * v_J[2][0] -
            focal.y * rz2 * v_J[1][1] + 2.0 * focal.y * p_view.y * rz3 * v_J[2][1]
    );

    v_mean += vec3f(dot(v_t, W[0]), dot(v_t, W[1]), dot(v_t, W[2]));

    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    // TODO: Is this definitely the same as above?
    let v_V_symm = mat3x3f(
        vec3f(
            v_cov3d0,
            0.5 * v_cov3d1,
            0.5 * v_cov3d2,
        ),
        vec3f(
            0.5 * v_cov3d1,
            v_cov3d3,
            0.5 * v_cov3d4,
        ),
        vec3f(
            0.5 * v_cov3d2,
            0.5 * v_cov3d4,
            v_cov3d5
        )
    );

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    let v_M = 2.0 * v_V_symm * M;

    let v_scale = vec3f(
        dot(R[0], v_M[0]),
        dot(R[1], v_M[1]),
        dot(R[2], v_M[2]),
    );

    let v_R = v_M * S;
    let v_quat = quat_to_rotmat_vjp(quat, v_R);

    v_quats[idx] = v_quat;
    v_scales[idx] = vec4f(v_scale, 0.0);
    v_means[idx] = vec4f(v_mean, 0.0f);
}