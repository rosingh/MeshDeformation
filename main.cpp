#include <igl/readOFF.h>
#include <igl/readCSV.h>
#include <igl/cotmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/per_vertex_normals.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

Eigen::MatrixXd smooth_mesh(Eigen::SparseMatrix<double> &A_fc, Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &A_free, const Eigen::Ref<const Eigen::MatrixXd> &vc);
Eigen::VectorXd elementwise_dot_sum(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

typedef struct Handle
{
  double x;
  double y;
  double z;
  double alpha;
  double beta;
  double gamma;
} Handle;

int main(int argc, char *argv[])
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::SparseMatrix<double> L;
  Eigen::SparseMatrix<double> M;

  // Load a mesh in OFF format
  igl::readOFF("../hand.off", V, F);
  Eigen::ArrayXXd v_array = V.array();
  v_array.rowwise() -= v_array.colwise().minCoeff();
  v_array /= v_array.maxCoeff();
  V = v_array.matrix();

  std::vector<std::vector<int>> adj_list;
  igl::adjacency_list(F, adj_list);

  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
  M = M.cwiseInverse();

  Eigen::MatrixXi lab;
  igl::readCSV("../hand.label.csv", lab);
  Eigen::ArrayXi labels = (lab.array()).cast<int>();
  Eigen::ArrayXi handles = (labels != 0).cast<int>();

  // Find free vertices
  std::vector<int> free_vertices;
  free_vertices.reserve(handles.size());
  for (unsigned i = 0; i < handles.size(); ++i)
  {
    if (handles[i] == 0)
    {
      free_vertices.push_back(i);
    }
  }

  // Find constrained vertices
  std::vector<int> constrained_vertices;
  constrained_vertices.reserve(handles.size());
  for (unsigned i = 0; i < handles.size(); ++i)
  {
    if (handles[i] != 0)
    {
      constrained_vertices.push_back(i);
    }
  }

  Eigen::SparseMatrix<double> A = L * M * L;

  // Construct A_free and A_fc
  Eigen::SparseMatrix<double> A_free(free_vertices.size(), free_vertices.size());
  Eigen::SparseMatrix<double> A_fc(free_vertices.size(), constrained_vertices.size());

  std::vector<Eigen::Triplet<double>> tripletList_free;
  std::vector<Eigen::Triplet<double>> tripletList_fc;
  tripletList_free.reserve(free_vertices.size() * free_vertices.size());
  tripletList_fc.reserve(free_vertices.size() * constrained_vertices.size());

  for (unsigned i = 0; i < free_vertices.size(); ++i)
  {
    for (unsigned j = 0; j < free_vertices.size(); ++j)
    {
      if (A.coeff(free_vertices[i], free_vertices[j]) != 0)
      {
        tripletList_free.push_back(Eigen::Triplet<double>(i, j, A.coeff(free_vertices[i], free_vertices[j])));
      }
    }
    for (unsigned j = 0; j < constrained_vertices.size(); ++j)
    {
      if (A.coeff(free_vertices[i], constrained_vertices[j]) != 0)
      {
        tripletList_fc.push_back(Eigen::Triplet<double>(i, j, A.coeff(free_vertices[i], constrained_vertices[j])));
      }
    }
  }

  A_free.setFromTriplets(tripletList_free.begin(), tripletList_free.end());
  A_fc.setFromTriplets(tripletList_fc.begin(), tripletList_fc.end());

  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(A_free);

  Eigen::MatrixXd v_free = smooth_mesh(A_fc, chol, V(constrained_vertices, Eigen::all));
  Eigen::MatrixXd displacements = V(free_vertices, Eigen::all) - v_free;

  Eigen::MatrixXd V_smooth(V);
  V_smooth(free_vertices, Eigen::all) = v_free;

  Eigen::MatrixXd normals;
  igl::per_vertex_normals(V_smooth, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, normals);

  std::vector<Eigen::VectorXd> xis;
  std::vector<int> proj_i_indices;
  std::vector<int> proj_j_indices;

  for (int i = 0; i < adj_list.size(); ++i)
  {
    if (handles[i] == 0)
    {
      std::vector<Eigen::VectorXd> projections;
      for (int j = 0; j < adj_list[i].size(); ++j)
      {
        int neighbor_index = adj_list[i][j];
        Eigen::VectorXd edge_vector = V_smooth.row(neighbor_index) - V_smooth.row(i);
        double dot_prod = edge_vector.dot(normals.row(i));
        Eigen::VectorXd projection = edge_vector - (dot_prod*normals.row(i).transpose());
        
        projections.push_back(projection);
        
      }
      // Find the longest projection
      Eigen::VectorXd lengths(projections.size());
      for (unsigned k = 0; k < projections.size(); ++k) {
        lengths[k] = projections[k].norm();
      }
      
      
      int j_star;
      lengths.maxCoeff(&j_star);
      unsigned chosen_neighbor_global_index = adj_list[i][j_star];
      proj_i_indices.push_back(i);
      proj_j_indices.push_back(chosen_neighbor_global_index);

      Eigen::VectorXd xi = projections[j_star] / projections[j_star].norm();
      xis.push_back(xi);
      
    }
  }
  Eigen::VectorXi proj_i_indices_eigen = Eigen::Map<Eigen::VectorXi>(proj_i_indices.data(), proj_i_indices.size());
  Eigen::VectorXi proj_j_indices_eigen = Eigen::Map<Eigen::VectorXi>(proj_j_indices.data(), proj_j_indices.size());

  Eigen::MatrixXd xis_mat(xis.size(), 3);
  for (size_t i = 0; i < xis.size(); ++i) {
    xis_mat.row(i) = xis[i];
  }
  Eigen::MatrixXd free_n(free_vertices.size(), 3);
  for (size_t i = 0; i < free_vertices.size(); ++i) {
    free_n.row(i) = normals.row(free_vertices[i]);
  }
  std::cout << "xis dimensions: " << xis_mat.rows() << "x" << xis_mat.cols() << std::endl;
  std::cout << "free_n dimensions: " << free_n.rows() << "x" << free_n.cols() << std::endl;
  


  Eigen::MatrixXd yis(free_vertices.size(), 3);
  for (size_t i = 0; i < free_vertices.size(); ++i)
  {
    Eigen::Vector3d free_n_vec = free_n.row(i);
    Eigen::Vector3d xis_vec = xis_mat.row(i);
    yis.row(i) = free_n_vec.cross(xis_vec);
  }

  Eigen::VectorXd dx = elementwise_dot_sum(displacements, xis_mat);
  Eigen::VectorXd dy = elementwise_dot_sum(displacements, yis);
  Eigen::VectorXd dn = elementwise_dot_sum(displacements, free_n);


  // Init the viewer
  igl::opengl::glfw::Viewer viewer;

  // Attach a menu plugin
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  plugin.widgets.push_back(&menu);

  // manipulate handles
  static int selectedHandle = 1;
  double doubleVariable = 0.1f;

  // Add content to the default menu window
  menu.callback_draw_viewer_menu = [&]()
  {
    // Draw parent menu content
    menu.draw_viewer_menu();

    // Add new group
    if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
    {
      // Expose variable directly ...
      ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

      // ... or using a custom callback
      static bool boolVariable = true;
      if (ImGui::Checkbox("bool", &boolVariable))
      {
        // do something
        std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
      }

      // Expose an enumeration type
      enum Orientation
      {
        Up = 0,
        Down,
        Left,
        Right
      };
      static Orientation dir = Up;
      ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

      // We can also use a std::vector<std::string> defined dynamically
      static int num_choices = 3;
      static std::vector<std::string> choices;
      static int idx_choice = 0;
      if (ImGui::InputInt("Num letters", &num_choices))
      {
        num_choices = std::max(1, std::min(26, num_choices));
      }
      if (num_choices != (int)choices.size())
      {
        choices.resize(num_choices);
        for (int i = 0; i < num_choices; ++i)
          choices[i] = std::string(1, 'A' + i);
        if (idx_choice >= num_choices)
          idx_choice = num_choices - 1;
      }
      ImGui::Combo("Letter", &idx_choice, choices);

      // Add a button
      if (ImGui::Button("Print Hello", ImVec2(-1, 0)))
      {
        std::cout << "Hello\n";
      }
    }
  };

  // Plot the mesh
  viewer.data().set_mesh(V_smooth, F);
  viewer.data().add_label(viewer.data().V.row(0) + viewer.data().V_normals.row(0).normalized() * 0.005, "Hello World!");
  viewer.launch();
}

Eigen::MatrixXd smooth_mesh(Eigen::SparseMatrix<double> &A_fc, Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &A_free, const Eigen::Ref<const Eigen::MatrixXd> &vc)
{
  std::cout << "A_fc dimensions: " << A_fc.rows() << "x" << A_fc.cols() << std::endl;
  std::cout << "vc dimensions: " << vc.rows() << "x" << vc.cols() << std::endl;
  auto A_fcv_c = A_fc * vc;
  std::cout << "A_fcv_c dimensions: " << A_fcv_c.rows() << "x" << A_fcv_c.cols() << std::endl;

  auto v_free = A_free.solve(-A_fcv_c);
  std::cout << "v_free dimensions: " << v_free.rows() << "x" << v_free.cols() << std::endl;
  return v_free;
}

Eigen::VectorXd elementwise_dot_sum(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
  return (a.array() * b.array()).rowwise().sum();
}