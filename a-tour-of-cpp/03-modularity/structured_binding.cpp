#include <iostream>

using namespace std;

struct Entry
{
    string name;
    int value;
};

int main()
{
    auto [n, v] = Entry{"name", 1};

    cout << "{ " << n << ", " << v << " }\n";
}
