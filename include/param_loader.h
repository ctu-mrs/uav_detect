#ifndef PARAM_LOADER_H
#define PARAM_LOADER_H

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <string>
#include <iostream>

bool load_successful = true;

template <typename T>
T load_param(ros::NodeHandle &nh, const std::string &name, const T &default_value, bool optional = true, bool print_value = true)
{
  T loaded = default_value;
  bool success = nh.getParam(name, loaded);
  if (success)
  {
    if (print_value)
      std::cout << "\t" << name << ":\t" << loaded << std::endl;
  } else
  {
    loaded = default_value;
    if (!optional)
    {
      ROS_ERROR("Could not load non-optional parameter %s", name.c_str());
      load_successful = false;
    } else
    {
      std::cout << "\t" << name << ":\t" << loaded << std::endl;
    }
  }
  return loaded;
}

template <typename T>
T load_param_compulsory(ros::NodeHandle &nh, const std::string &name, bool print_value = true)
{
  return load_param(nh, name, T(), false, print_value);
}

template <typename ConfigType>
class DynamicReconfigureMgr
{
  public:
    ConfigType config_latest;
    void dynamic_reconfigure_callback(ConfigType& config, uint32_t level)
    {
      config_latest = config;
      ROS_INFO("Dynamic reconfigure request received:");
      std::vector<typename ConfigType::AbstractParamDescriptionConstPtr> descrs = config.__getParamDescriptions__();
      for (auto &descr : descrs)
      {
        boost::any val;
        descr->getValue(config, val);
        std::cout << "\t" << descr->name << ":\t";
        // try to guess the correct value (these should be the only ones supported)
        int *intval;
        double *doubleval;
        bool *boolval;
        std::string *stringval;
        if ((intval = boost::any_cast<int>(&val)))
          std::cout << *intval << std::endl;
        else if ((doubleval = boost::any_cast<double>(&val)))
          std::cout << *doubleval << std::endl;
        else if ((boolval = boost::any_cast<bool>(&val)))
          std::cout << *boolval << std::endl;
        else if ((stringval = boost::any_cast<std::string>(&val)))
          std::cout << *stringval << std::endl;
        else
          std::cout << "unknown type" << std::endl;
      }
    };
};

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
