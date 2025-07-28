#include <iostream>
#include <limits>
#include <cstddef>

int main() {
    std::cout << "size_t max: " << std::numeric_limits<size_t>::max() << std::endl;
    std::cout << "unsigned int max: " << std::numeric_limits<unsigned int>::max() << std::endl;
    std::cout << "size_t bytes: " << sizeof(size_t) << " bytes" << std::endl;
    return 0;
}
