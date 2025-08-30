#include "../exercise.h"
#include <bits/types/cookie_io_functions_t.h>
#include <cstring>
// READ: 类模板 <https://cppreference.cn/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        for (int i = 0; i < 4; ++i) {
            shape[i] = shape_[i];
            size *= shape[i];
        }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
        // TODO: 实现单向广播的加法
        // 1. 检查广播兼容性
        for (int i = 0; i < 4; ++i) {
            if (others.shape[i] != 1 && others.shape[i] != this->shape[i]) {
                throw std::invalid_argument("不兼容的张量形状，无法广播");
            }
        }

        // 2. 计算步长 (将4D索引映射到线性内存的偏移量)
        unsigned int this_strides[4] = {1, 1, 1, 1};
        unsigned int others_strides[4] = {1, 1, 1, 1};

        // 计算this的步长 (从后往前乘)
        for (int i = 2; i >= 0; --i) {
            this_strides[i] = this_strides[i + 1] * this->shape[i + 1];
        }

        // 计算others的步长 (广播维度步长为0)
        for (int i = 2; i >= 0; --i) {
            others_strides[i] = others.shape[i + 1] == 1 ? 0 : others_strides[i + 1] * others.shape[i + 1];
        }

        // 3. 执行广播加法
        unsigned int total_elements = this_strides[0] * this->shape[0];
        for (unsigned int idx = 0; idx < total_elements; ++idx) {
            // 计算当前索引在4D空间中的坐标
            unsigned int coords[4];
            unsigned int temp = idx;
            for (int i = 0; i < 4; ++i) {
                coords[i] = temp / this_strides[i];
                temp %= this_strides[i];
            }

            // 计算others的对应线性索引
            unsigned int others_idx = 0;
            for (int i = 0; i < 4; ++i) {
                others_idx += coords[i] * others_strides[i];
            }

            // 元素相加
            this->data[idx] += others.data[others_idx];
        }

        return *this;
    }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D(shape, data);
        auto t1 = Tensor4D(shape, data);
        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D(s0, d0);
        auto t1 = Tensor4D(s1, d1);
        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
