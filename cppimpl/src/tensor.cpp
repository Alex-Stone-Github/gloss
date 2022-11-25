#include "tensor.hpp"


static size_t glt::compute_size(const glt::Shape& shape) {
    size_t product = 1;
    for (auto val : shape) {
        product *= val;
    }
    return product;
}
static size_t glt::compute_index(const Shape& shape, const Shape& index) {
    size_t total = 0;
}
