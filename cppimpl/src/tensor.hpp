#pragma once
#include <iostream>
#include <vector>

namespace glt {

using Shape = std::vector<size_t>;

static size_t compute_size(const Shape& shape);
static size_t compute_index(const Shape& shape, const Shape& index);

template <typename T>
class Tensor {
private:
    Shape shape;
    T* values;
public:
    Tensor(const Shape& shape);
};


}
