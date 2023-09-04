#include <iostream>

using namespace std;

double double_div(double x, double y)
{
    if (y == 0)
    {
        throw std::invalid_argument{"division by 0..."};
    }
    return x / y;
}

int main()
{
    try
    {
        double x = double_div(1, 0);

        cout << "I did the impossible! 1 / 0 = " << x;
    }
    catch (std::invalid_argument &err)
    {
        cerr << err.what();
    }
    catch (...)
    {
        cerr << "Unknown exception...";
    }
}