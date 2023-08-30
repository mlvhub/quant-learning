
class Vector
{
public:
    Vector(int s);
    double &operator[](int i);
    int size();
    Vector operator+(Vector &v);

private:
    double *elem;
    int sz;
};