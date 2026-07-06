#include <qnet/optimizer.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace cosq::qnet {

namespace {

constexpr uint32_t OPTIMIZER_MAGIC = 0x54504F51; // "QOPT" as LE uint32
constexpr uint32_t OPTIMIZER_VERSION = 1;

void write_u32(std::ofstream& out, uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint32_t read_u32(std::ifstream& in) {
    uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void write_u64(std::ofstream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint64_t read_u64(std::ifstream& in) {
    uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void write_float(std::ofstream& out, float value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

float read_float(std::ifstream& in) {
    float value = 0.0f;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void write_bool(std::ofstream& out, bool value) {
    uint8_t encoded = value ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&encoded), sizeof(encoded));
}

bool read_bool(std::ifstream& in) {
    uint8_t encoded = 0;
    in.read(reinterpret_cast<char*>(&encoded), sizeof(encoded));
    return encoded != 0;
}

void write_string(std::ofstream& out, const char* value) {
    std::string str(value);
    write_u32(out, static_cast<uint32_t>(str.size()));
    out.write(str.data(), static_cast<std::streamsize>(str.size()));
}

std::string read_string(std::ifstream& in) {
    uint32_t size = read_u32(in);
    std::string value(size, '\0');
    in.read(value.data(), static_cast<std::streamsize>(size));
    return value;
}

void write_tensor(std::ofstream& out, const Tensor& tensor) {
    write_u32(out, static_cast<uint32_t>(tensor.shape().size()));
    for (size_t dim : tensor.shape()) {
        write_u64(out, static_cast<uint64_t>(dim));
    }
    write_u64(out, static_cast<uint64_t>(tensor.size()));
    out.write(reinterpret_cast<const char*>(tensor.data()),
              static_cast<std::streamsize>(tensor.size() * sizeof(float)));
}

Tensor read_tensor(std::ifstream& in) {
    uint32_t ndim = read_u32(in);
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = static_cast<size_t>(read_u64(in));
    }
    uint64_t size = read_u64(in);
    std::vector<float> data(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return Tensor(shape, data);
}

void write_optimizer_header(std::ofstream& out,
                            const Optimizer& optimizer,
                            size_t parameter_count) {
    write_u32(out, OPTIMIZER_MAGIC);
    write_u32(out, OPTIMIZER_VERSION);
    write_string(out, optimizer.type_name());
    write_u64(out, static_cast<uint64_t>(parameter_count));
}

void validate_optimizer_header(std::ifstream& in,
                               const Optimizer& optimizer,
                               size_t parameter_count) {
    uint32_t magic = read_u32(in);
    if (magic != OPTIMIZER_MAGIC) {
        throw std::runtime_error("Invalid optimizer state file (bad magic)");
    }

    uint32_t version = read_u32(in);
    if (version > OPTIMIZER_VERSION) {
        throw std::runtime_error("Unsupported optimizer state version");
    }

    std::string type = read_string(in);
    if (type != optimizer.type_name()) {
        throw std::runtime_error("Optimizer type mismatch while loading state");
    }

    uint64_t saved_parameter_count = read_u64(in);
    if (saved_parameter_count != parameter_count) {
        throw std::runtime_error("Optimizer parameter count mismatch while loading state");
    }
}

} // namespace

Optimizer::Optimizer(std::vector<std::shared_ptr<Node>> parameters)
    : parameters_(std::move(parameters)) {
    validate_parameters(parameters_);
}

void Optimizer::validate_parameters(const std::vector<std::shared_ptr<Node>>& parameters) {
    for (const auto& parameter : parameters) {
        if (!parameter) {
            throw std::invalid_argument("optimizer parameter cannot be null");
        }
        if (parameter->op_type != OpType::PARAMETER) {
            throw std::invalid_argument("optimizer parameters must be PARAMETER nodes");
        }
    }
}

void Optimizer::zero_grad() {
    for (auto& parameter : parameters_) {
        parameter->gradient = Tensor();
    }
}

SGD::SGD(std::vector<std::shared_ptr<Node>> parameters,
         float learning_rate,
         float momentum,
         float weight_decay,
         bool nesterov)
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {
    if (learning_rate_ <= 0.0f) {
        throw std::invalid_argument("SGD learning_rate must be positive");
    }
    if (momentum_ < 0.0f) {
        throw std::invalid_argument("SGD momentum must be non-negative");
    }
    if (weight_decay_ < 0.0f) {
        throw std::invalid_argument("SGD weight_decay must be non-negative");
    }
    if (nesterov_ && momentum_ <= 0.0f) {
        throw std::invalid_argument("SGD nesterov requires momentum > 0");
    }
}

void SGD::step() {
    for (auto& parameter : parameters_) {
        if (parameter->gradient.size() == 0) {
            continue;
        }
        if (parameter->gradient.shape() != parameter->output.shape()) {
            throw std::logic_error("SGD: gradient shape mismatch for parameter");
        }

        Tensor grad = parameter->gradient.clone();
        for (size_t i = 0; i < grad.size(); ++i) {
            grad.data()[i] += weight_decay_ * parameter->output.data()[i];
        }

        if (momentum_ > 0.0f) {
            auto [it, inserted] = momentum_buffers_.emplace(parameter.get(),
                                                            Tensor(parameter->output.shape()));
            Tensor& buffer = it->second;
            for (size_t i = 0; i < grad.size(); ++i) {
                buffer.data()[i] = momentum_ * buffer.data()[i] + grad.data()[i];
                float update = nesterov_
                    ? grad.data()[i] + momentum_ * buffer.data()[i]
                    : buffer.data()[i];
                parameter->output.data()[i] -= learning_rate_ * update;
            }
            continue;
        }

        for (size_t i = 0; i < grad.size(); ++i) {
            parameter->output.data()[i] -= learning_rate_ * grad.data()[i];
        }
    }
}

const char* SGD::type_name() const {
    return "SGD";
}

void SGD::save_state(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open optimizer state file for writing: " + filepath);
    }

    write_optimizer_header(out, *this, parameters_.size());
    write_float(out, learning_rate_);
    write_float(out, momentum_);
    write_float(out, weight_decay_);
    write_bool(out, nesterov_);

    for (const auto& parameter : parameters_) {
        auto it = momentum_buffers_.find(parameter.get());
        bool has_buffer = it != momentum_buffers_.end();
        write_bool(out, has_buffer);
        if (has_buffer) {
            write_tensor(out, it->second);
        }
    }
}

void SGD::load_state(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open optimizer state file for reading: " + filepath);
    }

    validate_optimizer_header(in, *this, parameters_.size());
    learning_rate_ = read_float(in);
    momentum_ = read_float(in);
    weight_decay_ = read_float(in);
    nesterov_ = read_bool(in);

    if (learning_rate_ <= 0.0f || momentum_ < 0.0f || weight_decay_ < 0.0f ||
        (nesterov_ && momentum_ <= 0.0f)) {
        throw std::runtime_error("Invalid SGD state in checkpoint");
    }

    momentum_buffers_.clear();
    for (const auto& parameter : parameters_) {
        bool has_buffer = read_bool(in);
        if (!has_buffer) {
            continue;
        }

        Tensor buffer = read_tensor(in);
        if (buffer.shape() != parameter->output.shape()) {
            throw std::runtime_error("SGD momentum buffer shape mismatch");
        }
        momentum_buffers_.emplace(parameter.get(), std::move(buffer));
    }
}

Adam::Adam(std::vector<std::shared_ptr<Node>> parameters,
           float learning_rate,
           float beta1,
           float beta2,
           float epsilon,
           float weight_decay)
    : Optimizer(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay) {
    if (learning_rate_ <= 0.0f) {
        throw std::invalid_argument("Adam learning_rate must be positive");
    }
    if (beta1_ < 0.0f || beta1_ >= 1.0f || beta2_ < 0.0f || beta2_ >= 1.0f) {
        throw std::invalid_argument("Adam betas must be in [0, 1)");
    }
    if (epsilon_ <= 0.0f) {
        throw std::invalid_argument("Adam epsilon must be positive");
    }
    if (weight_decay_ < 0.0f) {
        throw std::invalid_argument("Adam weight_decay must be non-negative");
    }
}

float Adam::parameter_update(float m_hat, float v_hat) const {
    return learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
}

void Adam::step_impl(bool decoupled_weight_decay) {
    ++step_count_;
    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(step_count_));

    for (auto& parameter : parameters_) {
        if (parameter->gradient.size() == 0) {
            continue;
        }
        if (parameter->gradient.shape() != parameter->output.shape()) {
            throw std::logic_error("Adam/AdamW: gradient shape mismatch for parameter");
        }

        auto [m_it, m_inserted] = first_moment_.emplace(parameter.get(),
                                                        Tensor(parameter->output.shape()));
        auto [v_it, v_inserted] = second_moment_.emplace(parameter.get(),
                                                         Tensor(parameter->output.shape()));
        Tensor& m = m_it->second;
        Tensor& v = v_it->second;

        for (size_t i = 0; i < parameter->gradient.size(); ++i) {
            float grad = decoupled_weight_decay
                ? parameter->gradient.data()[i]
                : parameter->gradient.data()[i] + weight_decay_ * parameter->output.data()[i];

            m.data()[i] = beta1_ * m.data()[i] + (1.0f - beta1_) * grad;
            v.data()[i] = beta2_ * v.data()[i] + (1.0f - beta2_) * grad * grad;

            float m_hat = m.data()[i] / bias_correction1;
            float v_hat = v.data()[i] / bias_correction2;

            float update = parameter_update(m_hat, v_hat);
            if (decoupled_weight_decay) {
                float decayed = parameter->output.data()[i] *
                                (1.0f - learning_rate_ * weight_decay_);
                parameter->output.data()[i] = decayed - update;
            } else {
                parameter->output.data()[i] -= update;
            }
        }
    }
}

void Adam::step() {
    step_impl(false);
}

const char* Adam::type_name() const {
    return "Adam";
}

void Adam::save_state(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open optimizer state file for writing: " + filepath);
    }

    write_optimizer_header(out, *this, parameters_.size());
    write_float(out, learning_rate_);
    write_float(out, beta1_);
    write_float(out, beta2_);
    write_float(out, epsilon_);
    write_float(out, weight_decay_);
    write_u64(out, static_cast<uint64_t>(step_count_));

    for (const auto& parameter : parameters_) {
        auto m_it = first_moment_.find(parameter.get());
        auto v_it = second_moment_.find(parameter.get());
        bool has_state = m_it != first_moment_.end() && v_it != second_moment_.end();
        write_bool(out, has_state);
        if (has_state) {
            write_tensor(out, m_it->second);
            write_tensor(out, v_it->second);
        }
    }
}

void Adam::load_state(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open optimizer state file for reading: " + filepath);
    }

    validate_optimizer_header(in, *this, parameters_.size());
    learning_rate_ = read_float(in);
    beta1_ = read_float(in);
    beta2_ = read_float(in);
    epsilon_ = read_float(in);
    weight_decay_ = read_float(in);
    step_count_ = static_cast<size_t>(read_u64(in));

    if (learning_rate_ <= 0.0f || beta1_ < 0.0f || beta1_ >= 1.0f ||
        beta2_ < 0.0f || beta2_ >= 1.0f || epsilon_ <= 0.0f || weight_decay_ < 0.0f) {
        throw std::runtime_error("Invalid Adam state in checkpoint");
    }

    first_moment_.clear();
    second_moment_.clear();
    for (const auto& parameter : parameters_) {
        bool has_state = read_bool(in);
        if (!has_state) {
            continue;
        }

        Tensor first = read_tensor(in);
        Tensor second = read_tensor(in);
        if (first.shape() != parameter->output.shape() ||
            second.shape() != parameter->output.shape()) {
            throw std::runtime_error("Adam moment tensor shape mismatch");
        }
        first_moment_.emplace(parameter.get(), std::move(first));
        second_moment_.emplace(parameter.get(), std::move(second));
    }
}

AdamW::AdamW(std::vector<std::shared_ptr<Node>> parameters,
             float learning_rate,
             float beta1,
             float beta2,
             float epsilon,
             float weight_decay)
    : Adam(std::move(parameters), learning_rate, beta1, beta2, epsilon, weight_decay) {}

void AdamW::step() {
    step_impl(true);
}

const char* AdamW::type_name() const {
    return "AdamW";
}

} // namespace cosq::qnet
