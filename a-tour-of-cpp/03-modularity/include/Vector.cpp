#include "Vector.h"

Vector::Vector(int s) : elem{new double[s]}, sz{s} {}

double &Vector::operator[](int i) { return elem[i]; }

int Vector::size() { return sz; }

Vector Vector::operator+(Vector &v)
{
    Vector new_v(size() + v.size());

    int index = 0;
    for (int i = 0; i < size(); ++i)
    {
        new_v[index] = elem[i];
        ++index;
    }

    for (int i = 0; i < v.size(); ++i)
    {
        new_v[index] = v.elem[i];
        ++index;
    }

    return new_v;
}