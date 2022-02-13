#include <vector>

using namespace std;

class Solution
{
public:
	int maxArea(vector<int> &nums)
	{
		int mx = 0;
		size_t sz = nums.size();
		for (int i = 0; i < sz; ++i)
		{
			for (int j = i + 1; j < sz; ++j)
			{
				int area = (j - i) * min(nums[i], nums[j]);
				mx = max(mx, area);
			}
		}
		return mx;
	}

	int maxArea1(vector<int> &nums)
	{
		int i = 0, j = nums.size() - 1;
		int mx = 0;
		while (i < j)
		{
			int area = (j - i) * min(nums[i], nums[j]);
			mx = max(mx, area);
			if (nums[i] > nums[j])
			{
				j--;
			}
			else
			{
				i++;
			}
		}
		return mx;
	}
};