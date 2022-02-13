#include <vector>

using namespace std;

class Solution {
public:
	void moveZeroes(vector<int>& arr) {
		int j = 0;
		for (int i = 0; i < arr.size(); i++) {
			if (arr[i] != 0) {
				arr[j++] = arr[i];
			}
		}
		for (; j < arr.size(); j++) {
			arr[j] = 0;
		}
	}
};