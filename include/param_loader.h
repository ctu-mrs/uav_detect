#ifndef PARAM_LOADER_H
#define PARAM_LOADER_H

#include <ros/ros.h>
#include <string>
#include <iostream>

bool load_successful = true;

template <typename T> T load_param(ros::NodeHandle nh, std::string name, T default_value, bool print_value = true)
{
  T loaded;
  nh.param(name, loaded, default_value);
  if (print_value)
    std::cout << "\t" << name << ":\t" << loaded << std::endl;
  return loaded;
}

template <typename T> T load_param(ros::NodeHandle nh, std::string name, bool print_value = true)
{
  T loaded;
  bool success = nh.getParam(name, loaded);
  if (success)
  {
    if (print_value)
      std::cout << "\t" << name << ":\t" << loaded << std::endl;
  } else
  {
    ROS_ERROR("Could not load non-optional parameter %s", name.c_str());
    load_successful = false;
  }
  return loaded;
}

/* class LoadParam{ */
/* private: */
/*   ros::NodeHandle m_nh; */
/*   std::string m_name; */
/*   bool m_print_value; */
/* public: */
/*     LoadParam(ros::NodeHandle &nh, const std::string &name, bool print_value = true) */
/*       : m_nh(nh), m_name(name) {}; */
/*     /1* LoadParam(ros::NodeHandle &nh, const std::string &name, const T& default_value, bool print_value = true) *1/ */
/*     /1*   : m_nh(nh), m_name(name), m_print_value(print_value), m_default_value(default_value), m_use_default(true) {}; *1/ */

/*     template<typename T> */
/*     operator T() */
/*     { */
/*       /1* if (m_use_default) *1/ */
/*       /1* { *1/ */
/*       /1*   T loaded; *1/ */
/*       /1*   m_nh.param(m_name, &loaded, m_default_value); *1/ */
/*       /1*   if (m_print_value) *1/ */
/*       /1*     std::cout << "\t" << m_name << ":\t" << loaded << std::endl; *1/ */
/*       /1*   return loaded; *1/ */
/*       /1* } else *1/ */
/*       { */
/*         T loaded; */
/*         bool success = m_nh.getParam(m_name, &loaded); */
/*         if (success) */
/*         { */
/*           if (m_print_value) */
/*             std::cout << "\t" << m_name << ":\t" << loaded << std::endl; */
/*         } else */
/*         { */
/*           ROS_ERROR("Could not load non-optional parameter %s", m_name.c_str()); */
/*           /1* load_successful = false; *1/ */
/*         } */
/*         return loaded; */
/*       } */
/*     }; */

/* }; */

#endif // PARAM_LOADER_H
