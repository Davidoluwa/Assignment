#include <iostream>
using namespace std;

double toFahrenheit(double c) {
    return (c * 9.0 / 5.0) + 32.0;
}

int main() {
    double c;
    cout << "Enter Celsius: ";
    cin >> c;
    cout << c << "C = " << toFahrenheit(c) << "F" << endl;
    return 0;
}
