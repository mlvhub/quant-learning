#include <iostream>

using namespace std;

enum class Traffic_light
{
    green,
    yellow,
    red
};

Traffic_light &operator++(Traffic_light &t)
{
    using enum Traffic_light;

    switch (t)
    {
    case green:
        return t = yellow;
    case yellow:
        return t = red;
    case red:
        return t = green;
    }
}

std::string to_string(Traffic_light &t)
{
    using enum Traffic_light;

    switch (t)
    {
    case green:
        return "green";
    case yellow:
        return "yellow";
    case red:
        return "red";
    }
}

int main()
{
    auto tl = Traffic_light::green;
    for (int i = 0; i < 6; i++)
    {
        cout << to_string(tl) << '\n';
        ++tl;
    }
}
