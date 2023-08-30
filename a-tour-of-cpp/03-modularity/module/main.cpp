import Vector;
#include <iostream>

using namespace std;

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
