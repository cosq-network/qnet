#include <qnet/blas.hpp>
#include <qnet/ops.hpp>
#include <qnet/simd.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace cosq::qnet {

static void im2col(const float* input, size_t C, size_t H, size_t W,
                   size_t KH, size_t KW,
                   size_t stride_h, size_t stride_w,
                   size_t pad_h, size_t pad_w,
                   size_t OH, size_t OW,
                   float* col_buf) {
    size_t col_h = C * KH * KW;
    for (size_t oh = 0; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            float* col = col_buf + (oh * OW + ow) * col_h;
            size_t idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < KH; ++kh) {
                    for (size_t kw = 0; kw < KW; ++kw) {
                        int ih = static_cast<int>(oh * stride_h + kh) -
                                 static_cast<int>(pad_h);
                        int iw = static_cast<int>(ow * stride_w + kw) -
                                 static_cast<int>(pad_w);
                        if (ih >= 0 && ih < static_cast<int>(H) &&
                            iw >= 0 && iw < static_cast<int>(W)) {
                            col[idx] = input[c * H * W +
                                             static_cast<size_t>(ih) * W +
                                             static_cast<size_t>(iw)];
                        } else {
                            col[idx] = 0.0f;
                        }
                        ++idx;
                    }
                }
            }
        }
    }
}

static std::vector<size_t> broadcast_shape(const Tensor& a, const Tensor& b) {
    if (a.ndim() != b.ndim()) {
        throw std::invalid_argument("broadcast requires tensors with the same rank");
    }

    std::vector<size_t> shape(a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) {
        size_t ad = a.shape()[i];
        size_t bd = b.shape()[i];
        if (ad != bd && ad != 1 && bd != 1) {
            throw std::invalid_argument("tensors are not broadcast-compatible");
        }
        shape[i] = std::max(ad, bd);
    }
    return shape;
}

static size_t offset_for_index(const Tensor& tensor, const std::vector<size_t>& index) {
    size_t offset = tensor.offset();
    for (size_t dim = 0; dim < tensor.ndim(); ++dim) {
        size_t coordinate = tensor.shape()[dim] == 1 ? 0 : index[dim];
        offset += coordinate * tensor.strides()[dim];
    }
    return offset;
}

static std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& shape) {
    std::vector<size_t> index(shape.size(), 0);
    for (size_t dim = shape.size(); dim-- > 0;) {
        index[dim] = flat % shape[dim];
        flat /= shape[dim];
    }
    return index;
}

Tensor Ops::matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(
            "matmul shape mismatch: " + std::to_string(a.shape()[0]) + "x" +
            std::to_string(a.shape()[1]) + " vs " + std::to_string(b.shape()[0]) +
            "x" + std::to_string(b.shape()[1])
        );
    }

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    Tensor result({M, N});
    blas::sgemm(false, false, M, N, K, 1.0f,
                a.data(), K, b.data(), N, 0.0f,
                result.data(), N);
    return result;
}

Tensor Ops::add(const Tensor& a, const Tensor& b) {
    std::vector<size_t> result_shape = broadcast_shape(a, b);
    Tensor result(result_shape);

    if (a.shape() == b.shape()) {
        simd::add(result.data(), a.data(), b.data(), a.size());
        return result;
    }

    for (size_t i = 0; i < result.size(); ++i) {
        std::vector<size_t> index = unravel_index(i, result_shape);
        result.data()[i] = a.data()[offset_for_index(a, index)] +
                           b.data()[offset_for_index(b, index)];
    }
    return result;
}

Tensor Ops::mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("mul requires equal-shaped tensors");
    }

    Tensor result(a.shape());
    simd::mul(result.data(), a.data(), b.data(), a.size());
    return result;
}

Tensor Ops::relu(const Tensor& x) {
    Tensor result(x.shape());
    simd::relu(result.data(), x.data(), x.size());
    return result;
}

Tensor Ops::sigmoid(const Tensor& x) {
    Tensor result(x.shape());
    simd::sigmoid(result.data(), x.data(), x.size());
    return result;
}

Tensor Ops::softmax(const Tensor& x) {
    if (x.ndim() != 2) {
        throw std::invalid_argument("softmax requires 2D tensor (batch x logits)");
    }

    size_t batch = x.shape()[0];
    size_t dim = x.shape()[1];

    Tensor result(x.shape());
    for (size_t b = 0; b < batch; ++b) {
        float max_val = x.at({b, 0});
        for (size_t d = 1; d < dim; ++d) {
            max_val = std::max(max_val, x.at({b, d}));
        }

        float sum = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float v = std::exp(x.at({b, d}) - max_val);
            result.at({b, d}) = v;
            sum += v;
        }
        for (size_t d = 0; d < dim; ++d) {
            result.at({b, d}) /= sum;
        }
    }
    return result;
}

static size_t checked_class_index(float raw_index, size_t num_classes) {
    if (!std::isfinite(raw_index)) {
        throw std::invalid_argument("target index must be finite");
    }

    float rounded = std::round(raw_index);
    if (std::fabs(raw_index - rounded) > 1e-6f) {
        throw std::invalid_argument("target index must be an integer value");
    }

    if (rounded < 0.0f || rounded >= static_cast<float>(num_classes)) {
        throw std::out_of_range("target index out of bounds");
    }

    return static_cast<size_t>(rounded);
}

Tensor Ops::cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    if (logits.ndim() != 2) {
        throw std::invalid_argument("cross_entropy_loss requires 2D logits");
    }
    if (targets.ndim() != 1) {
        throw std::invalid_argument("cross_entropy_loss requires 1D targets");
    }
    if (logits.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("cross_entropy_loss batch size mismatch");
    }

    size_t batch = logits.shape()[0];
    size_t classes = logits.shape()[1];
    if (batch == 0 || classes == 0) {
        throw std::invalid_argument("cross_entropy_loss requires non-empty logits");
    }
    float loss = 0.0f;

    for (size_t b = 0; b < batch; ++b) {
        float max_logit = logits.at({b, 0});
        for (size_t c = 1; c < classes; ++c) {
            max_logit = std::max(max_logit, logits.at({b, c}));
        }

        float exp_sum = 0.0f;
        for (size_t c = 0; c < classes; ++c) {
            exp_sum += std::exp(logits.at({b, c}) - max_logit);
        }

        size_t target_index = checked_class_index(targets.at({b}), classes);
        float log_sum_exp = max_logit + std::log(exp_sum);
        loss += log_sum_exp - logits.at({b, target_index});
    }

    return Tensor({1}, {loss / static_cast<float>(batch)});
}

Tensor Ops::mse_loss(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::invalid_argument("mse_loss requires equal-shaped tensors");
    }
    if (pred.size() == 0) {
        throw std::invalid_argument("mse_loss requires non-empty tensors");
    }

    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float diff = pred.data()[i] - target.data()[i];
        loss += diff * diff;
    }

    return Tensor({1}, {loss / static_cast<float>(pred.size())});
}

Tensor Ops::binary_cross_entropy(const Tensor& pred, const Tensor& target) {
    if (pred.shape() != target.shape()) {
        throw std::invalid_argument("binary_cross_entropy requires equal-shaped tensors");
    }
    if (pred.size() == 0) {
        throw std::invalid_argument("binary_cross_entropy requires non-empty tensors");
    }

    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float x = pred.data()[i];
        float y = target.data()[i];
        loss += std::max(x, 0.0f) - x * y + std::log1p(std::exp(-std::fabs(x)));
    }

    return Tensor({1}, {loss / static_cast<float>(pred.size())});
}

Tensor Ops::conv2d(const Tensor& input, const Tensor& kernel,
                   size_t stride_h, size_t stride_w,
                   size_t pad_h, size_t pad_w) {
    if (input.ndim() != 4 || kernel.ndim() != 4) {
        throw std::invalid_argument("conv2d requires 4D tensors (NCHW)");
    }

    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];

    size_t K = kernel.shape()[0];
    size_t KC = kernel.shape()[1];
    size_t KH = kernel.shape()[2];
    size_t KW = kernel.shape()[3];

    if (KC != C) {
        throw std::invalid_argument("conv2d kernel input channels mismatch");
    }

    size_t OH = (H + 2 * pad_h - KH) / stride_h + 1;
    size_t OW = (W + 2 * pad_w - KW) / stride_w + 1;

    Tensor result({N, K, OH, OW});

    size_t col_h = C * KH * KW;
    size_t col_w = OH * OW;
    std::vector<float> col_buf(col_h * col_w);

    size_t kernel_flat_rows = K;
    size_t kernel_flat_cols = col_h;

    for (size_t n = 0; n < N; ++n) {
        im2col(input.data() + n * C * H * W,
               C, H, W, KH, KW, stride_h, stride_w, pad_h, pad_w, OH, OW,
               col_buf.data());

        float* out_n = result.data() + n * K * OH * OW;
        blas::sgemm(false, false,
                    kernel_flat_rows, col_w, kernel_flat_cols,
                    1.0f,
                    kernel.data(), kernel_flat_cols,
                    col_buf.data(), col_w,
                    0.0f,
                    out_n, col_w);
    }

    return result;
}

Tensor Ops::embedding(const Tensor& weight, const Tensor& indices) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("embedding weight must be 2D");
    }
    if (indices.ndim() != 1) {
        throw std::invalid_argument("embedding indices must be 1D");
    }

    size_t vocab = weight.shape()[0];
    size_t dim = weight.shape()[1];

    Tensor result({indices.shape()[0], dim});
    for (size_t i = 0; i < indices.shape()[0]; ++i) {
        size_t idx = static_cast<size_t>(indices.data()[i]);
        if (idx >= vocab) {
            throw std::out_of_range("embedding index out of bounds");
        }
        for (size_t d = 0; d < dim; ++d) {
            result.at({i, d}) = weight.at({idx, d});
        }
    }
    return result;
}

void Ops::matmul_backward(const Tensor& a, const Tensor& b,
                           const Tensor& grad_output,
                           Tensor& grad_a, Tensor& grad_b) {
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (grad_a.size() > 0) {
        std::memset(grad_a.data(), 0, grad_a.bytes());
        blas::sgemm(false, true, M, K, N, 1.0f,
                    grad_output.data(), N,
                    b.data(), N, 1.0f,
                    grad_a.data(), K);
    }

    if (grad_b.size() > 0) {
        std::memset(grad_b.data(), 0, grad_b.bytes());
        blas::sgemm(true, false, K, N, M, 1.0f,
                    a.data(), K,
                    grad_output.data(), N, 1.0f,
                    grad_b.data(), N);
    }
}

static void col2im(const float* dcol_buf, size_t C, size_t H, size_t W,
                   size_t KH, size_t KW,
                   size_t stride_h, size_t stride_w,
                   size_t pad_h, size_t pad_w,
                   size_t OH, size_t OW,
                   float* grad_input) {
    std::memset(grad_input, 0, C * H * W * sizeof(float));
    size_t col_h = C * KH * KW;
    for (size_t oh = 0; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < KH; ++kh) {
                    for (size_t kw = 0; kw < KW; ++kw) {
                        int ih = static_cast<int>(oh * stride_h + kh) -
                                 static_cast<int>(pad_h);
                        int iw = static_cast<int>(ow * stride_w + kw) -
                                 static_cast<int>(pad_w);
                        if (ih >= 0 && ih < static_cast<int>(H) &&
                            iw >= 0 && iw < static_cast<int>(W)) {
                            size_t col_idx =
                                (oh * OW + ow) * col_h +
                                c * KH * KW + kh * KW + kw;
                            size_t inp_idx =
                                c * H * W +
                                static_cast<size_t>(ih) * W +
                                static_cast<size_t>(iw);
                            grad_input[inp_idx] += dcol_buf[col_idx];
                        }
                    }
                }
            }
        }
    }
}

void Ops::relu_backward(const Tensor& dy, const Tensor& x,
                        Tensor& grad_x) {
    size_t n = dy.size();
    for (size_t i = 0; i < n; ++i) {
        grad_x.data()[i] = x.data()[i] > 0.0f ? dy.data()[i] : 0.0f;
    }
}

void Ops::add_backward(const Tensor& dy, const Tensor& a, const Tensor& b,
                       Tensor& grad_a, Tensor& grad_b) {
    if (dy.shape() != broadcast_shape(a, b)) {
        throw std::invalid_argument("add_backward gradient shape mismatch");
    }

    if (a.shape() == b.shape()) {
        if (grad_a.size() > 0) {
            std::memcpy(grad_a.data(), dy.data(), dy.bytes());
        }
        if (grad_b.size() > 0) {
            std::memcpy(grad_b.data(), dy.data(), dy.bytes());
        }
        return;
    }

    if (grad_a.size() > 0) {
        std::memset(grad_a.data(), 0, grad_a.bytes());
    }
    if (grad_b.size() > 0) {
        std::memset(grad_b.data(), 0, grad_b.bytes());
    }

    for (size_t i = 0; i < dy.size(); ++i) {
        std::vector<size_t> index = unravel_index(i, dy.shape());
        if (grad_a.size() > 0) {
            grad_a.data()[offset_for_index(a, index)] += dy.data()[i];
        }
        if (grad_b.size() > 0) {
            grad_b.data()[offset_for_index(b, index)] += dy.data()[i];
        }
    }
}

void Ops::mul_backward(const Tensor& dy, const Tensor& a, const Tensor& b,
                       Tensor& grad_a, Tensor& grad_b) {
    size_t n = dy.size();
    if (grad_a.size() > 0) {
        for (size_t i = 0; i < n; ++i) {
            grad_a.data()[i] = dy.data()[i] * b.data()[i];
        }
    }
    if (grad_b.size() > 0) {
        for (size_t i = 0; i < n; ++i) {
            grad_b.data()[i] = dy.data()[i] * a.data()[i];
        }
    }
}

void Ops::sigmoid_backward(const Tensor& dy, const Tensor& out,
                           Tensor& grad_x) {
    size_t n = dy.size();
    for (size_t i = 0; i < n; ++i) {
        float s = out.data()[i];
        grad_x.data()[i] = dy.data()[i] * s * (1.0f - s);
    }
}

void Ops::softmax_backward(const Tensor& dy, const Tensor& out,
                           Tensor& grad_x) {
    size_t batch = out.shape()[0];
    size_t dim = out.shape()[1];

    for (size_t b = 0; b < batch; ++b) {
        float sum_s_dy = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            sum_s_dy += out.at({b, d}) * dy.at({b, d});
        }
        for (size_t d = 0; d < dim; ++d) {
            float s = out.at({b, d});
            grad_x.at({b, d}) = s * (dy.at({b, d}) - sum_s_dy);
        }
    }
}

void Ops::cross_entropy_backward(const Tensor& logits, const Tensor& targets,
                                 const Tensor& grad_output,
                                 Tensor& grad_logits) {
    if (grad_logits.shape() != logits.shape()) {
        throw std::invalid_argument("cross_entropy_backward gradient shape mismatch");
    }

    size_t batch = logits.shape()[0];
    size_t classes = logits.shape()[1];
    if (batch == 0 || classes == 0) {
        throw std::invalid_argument("cross_entropy_backward requires non-empty logits");
    }

    Tensor probs = softmax(logits);
    float scale = grad_output.data()[0] / static_cast<float>(batch);

    for (size_t b = 0; b < batch; ++b) {
        size_t target_index = checked_class_index(targets.at({b}), classes);
        for (size_t c = 0; c < classes; ++c) {
            float grad = probs.at({b, c});
            if (c == target_index) {
                grad -= 1.0f;
            }
            grad_logits.at({b, c}) = grad * scale;
        }
    }
}

void Ops::mse_backward(const Tensor& pred, const Tensor& target,
                       const Tensor& grad_output,
                       Tensor& grad_pred, Tensor& grad_target) {
    if (grad_pred.shape() != pred.shape() || grad_target.shape() != target.shape()) {
        throw std::invalid_argument("mse_backward gradient shape mismatch");
    }
    if (pred.size() == 0) {
        throw std::invalid_argument("mse_backward requires non-empty tensors");
    }

    float scale = 2.0f * grad_output.data()[0] / static_cast<float>(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        float grad = scale * (pred.data()[i] - target.data()[i]);
        grad_pred.data()[i] = grad;
        grad_target.data()[i] = -grad;
    }
}

void Ops::bce_backward(const Tensor& pred, const Tensor& target,
                       const Tensor& grad_output,
                       Tensor& grad_pred, Tensor& grad_target) {
    if (grad_pred.shape() != pred.shape() || grad_target.shape() != target.shape()) {
        throw std::invalid_argument("bce_backward gradient shape mismatch");
    }
    if (pred.size() == 0) {
        throw std::invalid_argument("bce_backward requires non-empty tensors");
    }

    float scale = grad_output.data()[0] / static_cast<float>(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        float sigmoid_x = 1.0f / (1.0f + std::exp(-pred.data()[i]));
        grad_pred.data()[i] = scale * (sigmoid_x - target.data()[i]);
        grad_target.data()[i] = -scale * pred.data()[i];
    }
}

void Ops::conv2d_backward(const Tensor& dy, const Tensor& input,
                          const Tensor& kernel,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w,
                          Tensor& grad_input, Tensor& grad_kernel) {
    size_t N = input.shape()[0];
    size_t C = input.shape()[1];
    size_t H = input.shape()[2];
    size_t W = input.shape()[3];
    size_t K = kernel.shape()[0];
    size_t KH = kernel.shape()[2];
    size_t KW = kernel.shape()[3];

    size_t OH = (H + 2 * pad_h - KH) / stride_h + 1;
    size_t OW = (W + 2 * pad_w - KW) / stride_w + 1;

    size_t col_h = C * KH * KW;
    size_t col_w = OH * OW;

    if (grad_kernel.size() > 0) {
        std::memset(grad_kernel.data(), 0, grad_kernel.bytes());
        std::vector<float> col_buf(col_h * col_w);

        for (size_t n = 0; n < N; ++n) {
            im2col(input.data() + n * C * H * W,
                   C, H, W, KH, KW, stride_h, stride_w, pad_h, pad_w, OH, OW,
                   col_buf.data());

            const float* dy_n = dy.data() + n * K * OH * OW;
            blas::sgemm(false, true, K, col_h, col_w, 1.0f,
                        dy_n, col_w, col_buf.data(), col_w, 1.0f,
                        grad_kernel.data(), col_h);
        }
    }

    if (grad_input.size() > 0) {
        std::memset(grad_input.data(), 0, grad_input.bytes());
        std::vector<float> dcol_buf(col_h * col_w);

        for (size_t n = 0; n < N; ++n) {
            const float* dy_n = dy.data() + n * K * OH * OW;

            blas::sgemm(true, false, col_h, col_w, K, 1.0f,
                        kernel.data(), col_h, dy_n, col_w, 0.0f,
                        dcol_buf.data(), col_w);

            col2im(dcol_buf.data(), C, H, W, KH, KW,
                   stride_h, stride_w, pad_h, pad_w, OH, OW,
                   grad_input.data() + n * C * H * W);
        }
    }
}

void Ops::embedding_backward(const Tensor& dy, const Tensor& indices,
                             Tensor& grad_weight) {
    size_t dim = dy.shape()[1];
    for (size_t i = 0; i < indices.shape()[0]; ++i) {
        size_t idx = static_cast<size_t>(indices.data()[i]);
        for (size_t d = 0; d < dim; ++d) {
            grad_weight.at({idx, d}) += dy.at({i, d});
        }
    }
}

Tensor Ops::dropout_forward(const Tensor& x, float rate, Tensor& mask) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("dropout rate must be in [0, 1)");
    }

    mask = Tensor(x.shape());
    float scale = 1.0f / (1.0f - rate);
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        mask.data()[i] = (dist(rng) >= rate) ? scale : 0.0f;
        result.data()[i] = x.data()[i] * mask.data()[i];
    }
    return result;
}

void Ops::dropout_backward(const Tensor& dy, const Tensor& mask,
                           Tensor& grad_x) {
    if (dy.shape() != mask.shape()) {
        throw std::invalid_argument("dropout_backward shape mismatch");
    }
    for (size_t i = 0; i < dy.size(); ++i) {
        grad_x.data()[i] = dy.data()[i] * mask.data()[i];
    }
}

Tensor Ops::layer_norm(const Tensor& x, const Tensor* gamma,
                       const Tensor* beta, float eps) {
    if (x.ndim() < 1) {
        throw std::invalid_argument("layer_norm requires at least 1D input");
    }
    size_t N = x.shape().back();
    size_t batch = x.size() / N;

    Tensor result(x.shape());
    for (size_t b = 0; b < batch; ++b) {
        float mean = 0.0f;
        for (size_t i = 0; i < N; ++i) mean += x.data()[b * N + i];
        mean /= static_cast<float>(N);

        float var = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = x.data()[b * N + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(N);

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < N; ++i) {
            float y = (x.data()[b * N + i] - mean) * inv_std;
            if (gamma) y *= gamma->data()[i];
            if (beta) y += beta->data()[i];
            result.data()[b * N + i] = y;
        }
    }
    return result;
}

void Ops::layer_norm_backward(const Tensor& dy, const Tensor& x,
                              const Tensor* gamma, const Tensor* beta,
                              float eps, Tensor& grad_x,
                              Tensor* grad_gamma, Tensor* grad_beta) {
    (void)beta;
    size_t N = x.shape().back();
    size_t batch = x.size() / N;

    std::vector<float> means(batch), inv_stds(batch);
    for (size_t b = 0; b < batch; ++b) {
        float mean = 0.0f;
        for (size_t i = 0; i < N; ++i) mean += x.data()[b * N + i];
        mean /= static_cast<float>(N);
        means[b] = mean;

        float var = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = x.data()[b * N + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(N);
        inv_stds[b] = 1.0f / std::sqrt(var + eps);
    }

    for (size_t b = 0; b < batch; ++b) {
        float inv_std = inv_stds[b];
        float mean_dy = 0.0f, mean_dy_yhat = 0.0f;

        for (size_t i = 0; i < N; ++i) {
            float y_hat = (x.data()[b * N + i] - means[b]) * inv_std;
            float w = gamma ? gamma->data()[i] : 1.0f;
            mean_dy += dy.data()[b * N + i];
            mean_dy_yhat += dy.data()[b * N + i] * y_hat;
        }
        mean_dy /= static_cast<float>(N);
        mean_dy_yhat /= static_cast<float>(N);

        for (size_t i = 0; i < N; ++i) {
            float y_hat = (x.data()[b * N + i] - means[b]) * inv_std;
            float w = gamma ? gamma->data()[i] : 1.0f;
            float dx = (dy.data()[b * N + i] - mean_dy - y_hat * mean_dy_yhat) * inv_std * w;
            grad_x.data()[b * N + i] = dx;

            if (grad_gamma) {
                grad_gamma->data()[i] += dy.data()[b * N + i] * y_hat;
            }
            if (grad_beta) {
                grad_beta->data()[i] += dy.data()[b * N + i];
            }
        }
    }
}

} // namespace cosq::qnet
