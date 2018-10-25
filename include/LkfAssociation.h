#include "main.h"
#include "mrs_lib/Lkf.h"

namespace uav_detect
{
  class LkfAssociation : public mrs_lib::Lkf
  {
    public:
      LkfAssociation(const int id,
                const int n, const int m, const int p,
                const Eigen::MatrixXd A, const Eigen::MatrixXd B,
                const Eigen::MatrixXd R, const Eigen::MatrixXd Q,
                const Eigen::MatrixXd P)
        : Lkf(n, m, p, A, B, R, Q, P), id(id), m_n_corrections(0)
      {};

      const int id;

      virtual Eigen::VectorXd doCorrection(void)
      {
        m_n_corrections++;
        return Lkf::doCorrection();
      }

      int getNCorrections(void) const
      {
        return m_n_corrections;
      }

    protected:
      int m_n_corrections;
  };
}
