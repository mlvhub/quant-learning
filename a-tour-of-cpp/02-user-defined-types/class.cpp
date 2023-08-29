#include <iostream>

using namespace std;

class Vector
{
public:
    // constructor
    Vector(int s) : elem{new double[s]}, sz{s} {}
    // subscript operation
    double &operator[](int i) { return elem[i]; }
    int size() { return sz; }
    Vector operator+(Vector &v)
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

private:
    double *elem;
    int sz;
};

int main()
{
    int s = 10;
    Vector v(s);

    for (int i = 0; i < s; ++i)
        v[i] = i;

    for (int i = 0; i < v.size(); ++i)
        cout << v[i] << '\n';

    int s2 = 6;
    Vector v2(s2);
    for (int i = 0; i < s2; ++i)
        v2[i] = i;

    Vector v3 = v + v2;
    cout << "Vector Sum\n";
    for (int i = 0; i < v3.size(); ++i)
        cout << v3[i] << '\n';
}
