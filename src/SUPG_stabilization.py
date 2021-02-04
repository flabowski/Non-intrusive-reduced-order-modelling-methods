import numpy as np
import dolfin as df
import matplotlib.pyplot as plt

mesh = df.RectangleMesh(df.Point(.0, .0), df.Point(2.2, 0.41), 5, 5)
V = df.VectorFunctionSpace(mesh, 'P', 2)
Q = df.FunctionSpace(mesh, 'P', 1)
u_ = df.Function(V)
r_ = df.Function(Q)
u_.vector().vec().array = np.random.rand(*u_.vector().vec().array.shape)


# https://fenicsproject.discourse.group/t/pass-function-to-c-expression/1081
tau_code = '''
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
namespace py = pybind11;

class SUPG : public dolfin::Expression
{
    public:
        double D;
        std::shared_ptr<dolfin::Function> velocity;

        SUPG(std::shared_ptr<dolfin::Function> u_) : dolfin::Expression(){
            velocity = u_;
        }

        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x,
                  const ufc::cell& c) const override{
            std::shared_ptr<const dolfin::Mesh> mesh = velocity->function_space()->mesh();
            dolfin::Cell cell(*mesh, c.index);
            velocity->eval(values, x);  // values now holds the velocity
            double tau = 0.0;
            double h = cell.h();
            double magnitude = 0.0;  // pythagoras
            for (uint i=0; i<values.size(); ++i){
                magnitude += values[i] * values[i];
            }
            magnitude = sqrt(magnitude);
            double Pe = magnitude * h / (2.0 * D);
            if (Pe > DOLFIN_EPS){
                tau = h / (2.0*magnitude) * (1.0/tanh(Pe) - 1.0/Pe);
            }
            values[0] = tau;
        };

};


PYBIND11_MODULE(SIGNATURE, m){
    py::class_<SUPG, std::shared_ptr<SUPG>, dolfin::Expression>
    (m, "SUPG").def(py::init<std::shared_ptr<dolfin::Function>>())
    .def("eval", &SUPG::eval)
    .def_readwrite("D", &SUPG::D);
}

'''

tau_code_compiled = df.compile_cpp_code(tau_code)
tau_c = df.CompiledExpression(tau_code_compiled.SUPG(u_.cpp_object()),
                              D=.1, element=r_.ufl_element())

# tau_c.value_rank()
values = np.zeros(2)
x = np.array([0.5, 0.5])
tau_c.eval(values, x)

print(df.assemble(tau_c*df.dx(domain=mesh)))
tau = df.interpolate(tau_c, Q)

df.plot(tau)
plt.show()
