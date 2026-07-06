#include <qnet/optimizer.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace cosq::qnet {

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

} // namespace cosq::qnet
