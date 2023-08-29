#include <iostream>

using namespace std;

/*
    Initialization:
    - Pointer: points to object
    - Reference: refers to object

    Assignment:
    - Pointer: points to new object
    - Reference: assigns new value to the referenced object
*/
int main()
{
    int x = 2;
    int y = 3;

    cout << "Pointers:\n";
    int *p1 = &x;
    int *p2 = &y;
    cout << *p1 << '-' << *p2 << '\n';

    p1 = p2;
    cout << *p1 << '-' << *p2 << '\n';
    cout << x << '-' << y << '\n';

    cout << "References:\n";
    int &r1 = x;
    int &r2 = y;
    cout << r1 << '-' << r2 << '\n';

    // `r1` will still reference `x`, but it will assign the value of `r2` to the underlying reference (`x`)
    r1 = r2;
    cout << r1 << '-' << r2 << '\n';
    cout << x << '-' << y << '\n';
}