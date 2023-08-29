#include <iostream>

using namespace std;

int count_x(const char *p, char x)
// count the number of occurrences of x in p[]
// p is assumed to point to a zero-terminated array of char (or to nothing)
{
    if (p == nullptr)
        return 0;
    int count = 0;
    while (*p)
    {
        if (*p == x)
            ++count;
        ++p;
    }
    return count;
}

int main()
{
    char p[] = {'a', 'b', 'a'};

    cout << count_x(p, 'a');
}