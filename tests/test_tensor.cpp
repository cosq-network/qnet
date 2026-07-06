#include <qnet/tensor.hpp>
#include <qnet/ops.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

using namespace cosq::qnet;

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

static void test_construction() {
    Tensor t({2, 3});
    assert(t.ndim() == 2);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.size() == 6);
    assert(t.strides()[0] == 3);
    assert(t.strides()[1] == 1);

    Tensor t2({2, 3}, {1, 2, 3, 4, 5, 6});
    assert(approx(t2.at({0, 0}), 1.0f));
    assert(approx(t2.at({0, 2}), 3.0f));
    assert(approx(t2.at({1, 1}), 5.0f));

    std::cout << "  [PASS] test_construction\n";
}

static void test_at() {
    Tensor t({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            t.at({i, j}) = static_cast<float>(i * 4 + j);

    assert(approx(t.at({0, 0}), 0.0f));
    assert(approx(t.at({1, 2}), 6.0f));
    assert(approx(t.at({2, 3}), 11.0f));

    std::cout << "  [PASS] test_at\n";
}

static void test_slice() {
    Tensor t({3, 4});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            t.at({i, j}) = static_cast<float>(i * 4 + j);

    auto s = t.slice(0, 1, 3);
    assert(s.shape()[0] == 2);
    assert(s.shape()[1] == 4);
    assert(approx(s.at({0, 0}), 4.0f));
    assert(approx(s.at({1, 3}), 11.0f));
    assert(s.is_view());

    std::cout << "  [PASS] test_slice\n";
}

static void test_clone() {
    Tensor t({2, 2}, {1, 2, 3, 4});
    auto c = t.clone();
    assert(!c.is_view());
    assert(approx(c.at({0, 0}), 1.0f));
    c.at({0, 0}) = 99.0f;
    assert(approx(t.at({0, 0}), 1.0f));

    std::cout << "  [PASS] test_clone\n";
}

static void test_view() {
    Tensor t({2, 6});
    for (size_t i = 0; i < t.size(); ++i)
        t.data()[i] = static_cast<float>(i);

    auto v = t.view({3, 4});
    assert(v.shape()[0] == 3);
    assert(v.shape()[1] == 4);
    assert(approx(v.at({0, 0}), 0.0f));
    assert(approx(v.at({2, 3}), 11.0f));

    std::cout << "  [PASS] test_view\n";
}

static void test_grad() {
    Tensor t({2, 2});
    assert(!t.has_grad());
    t.grad();
    assert(t.has_grad());
    t.zero_grad();
    assert(approx(t.grad().data()[0], 0.0f));

    std::cout << "  [PASS] test_grad\n";
}

static void test_ops() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});

    auto c = Ops::matmul(a, b);
    assert(c.shape()[0] == 2);
    assert(c.shape()[1] == 2);
    assert(approx(c.at({0, 0}), 58.0f));
    assert(approx(c.at({0, 1}), 64.0f));
    assert(approx(c.at({1, 0}), 139.0f));
    assert(approx(c.at({1, 1}), 154.0f));

    Tensor d({2, 2}, {1, 2, 3, 4});
    auto e = Ops::add(c, d);
    assert(approx(e.at({0, 0}), 59.0f));

    auto r = Ops::relu(Tensor({3}, {-1, 0, 2}));
    assert(approx(r.at({0}), 0.0f));
    assert(approx(r.at({1}), 0.0f));
    assert(approx(r.at({2}), 2.0f));

    std::cout << "  [PASS] test_ops\n";
}

int main() {
    std::cout << "Tensor tests:\n";
    test_construction();
    test_at();
    test_slice();
    test_clone();
    test_view();
    test_grad();
    test_ops();
    std::cout << "All tensor tests passed!\n";
    return 0;
}
