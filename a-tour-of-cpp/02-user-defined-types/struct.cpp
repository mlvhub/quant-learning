#include <iostream>

using namespace std;

struct Vector
{
    double *elem;
    int sz;
};

void vector_init(Vector &v, int s)
{
    v.elem = new double[s];
    v.sz = s;
}

int main()
{
    int s = 10;
    Vector v;
    vector_init(v, s);

    for (int i = 0; i < s; ++i)
        v.elem[i] = i;

    for (int i = 0; i < v.sz; ++i)
        cout << v.elem[i] << '\n';
}
