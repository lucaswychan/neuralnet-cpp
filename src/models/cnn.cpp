#include "cnn.hpp"

void CNN::apply_to_children(const function<void(Module&)>& fn)
{
    for (Module* layer : this->layers_)
    {
        fn(*layer);
    }
}
