#pragma once

#include <qnet/graph.hpp>

#include <cstddef>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cosq::qnet {

class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<Node>> parameters);
    virtual ~Optimizer() = default;

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    Optimizer(Optimizer&&) = default;
    Optimizer& operator=(Optimizer&&) = default;

    virtual void step() = 0;
    virtual void zero_grad();
    virtual const char* type_name() const = 0;
    virtual void save_state(const std::string& filepath) const = 0;
    virtual void load_state(const std::string& filepath) = 0;

    const std::vector<std::shared_ptr<Node>>& parameters() const { return parameters_; }

protected:
    std::vector<std::shared_ptr<Node>> parameters_;

    static void validate_parameters(const std::vector<std::shared_ptr<Node>>& parameters);
};

class SGD : public Optimizer {
public:
    SGD(std::vector<std::shared_ptr<Node>> parameters,
        float learning_rate,
        float momentum = 0.0f,
        float weight_decay = 0.0f,
        bool nesterov = false);

    void step() override;
    const char* type_name() const override;
    void save_state(const std::string& filepath) const override;
    void load_state(const std::string& filepath) override;

private:
    float learning_rate_;
    float momentum_;
    float weight_decay_;
    bool nesterov_;
    std::unordered_map<Node*, Tensor> momentum_buffers_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<std::shared_ptr<Node>> parameters,
         float learning_rate = 1e-3f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float epsilon = 1e-8f,
         float weight_decay = 0.0f);

    void step() override;
    const char* type_name() const override;
    void save_state(const std::string& filepath) const override;
    void load_state(const std::string& filepath) override;

protected:
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    size_t step_count_ = 0;
    std::unordered_map<Node*, Tensor> first_moment_;
    std::unordered_map<Node*, Tensor> second_moment_;

    virtual float parameter_update(float m_hat, float v_hat) const;

    void step_impl(bool decoupled_weight_decay);
};

class AdamW : public Adam {
public:
    AdamW(std::vector<std::shared_ptr<Node>> parameters,
          float learning_rate = 1e-3f,
          float beta1 = 0.9f,
          float beta2 = 0.999f,
          float epsilon = 1e-8f,
          float weight_decay = 0.01f);

    void step() override;
    const char* type_name() const override;
};

} // namespace cosq::qnet
