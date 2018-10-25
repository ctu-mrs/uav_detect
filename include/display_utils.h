#include <ros/ros.h>
#include <list>

template <typename T>
void add_to_buffer(T img, std::list<T>& bfr)
{
  bfr.push_back(img);
  if (bfr.size() > 100)
    bfr.pop_front();
}

template <class T>
T find_closest(ros::Time stamp, std::list<T>& bfr)
{
  T closest;
  double closest_diff;
  bool closest_set = false;

  for (auto& imptr : bfr)
  {
    double cur_diff = abs((imptr->header.stamp - stamp).toSec());

    if (!closest_set || cur_diff < closest_diff)
    {
      closest = imptr;
      closest_diff = cur_diff;
      closest_set = true;
    }
  }
  return closest;
}

