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

Eigen::MatrixXd smooth_mesh(const Eigen::SparseMatrix<double> &A_fc, const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &A_free, const Eigen::Ref<const Eigen::MatrixXd> &vc);
Eigen::VectorXd elementwise_dot_sum(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);
void move_handles(Eigen::MatrixXd &target_pos, const Eigen::MatrixXd &V, Eigen::ArrayXi &labels, int s, double x, double y, double z, double alpha, double beta, double gamma);
void position_deformer(Eigen::MatrixXd &target_pos, const Eigen::MatrixXi &F, Eigen::VectorXi &proj_i_indices, Eigen::VectorXi &proj_j_indices,
											 const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &A_free, const Eigen::SparseMatrix<double> &A_fc,
											 std::vector<int> &free_vertices, std::vector<int> &constrained_vertices,
											 const Eigen::MatrixXd &displace_components);
void assign_colors(const Eigen::ArrayXi &labels, Eigen::MatrixXd &C);

struct Segment
{
	float x;
	float y;
	float z;
	float alpha;
	float beta;
	float gamma;

	Segment() : x(0), y(0), z(0), alpha(0), beta(0), gamma(0) {}
};

int main(int argc, char *argv[])
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::SparseMatrix<double> L;
	Eigen::SparseMatrix<double> M;

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
	Eigen::MatrixXd colors;
	assign_colors(labels, colors);

	std::vector<int> free_vertices;
	free_vertices.reserve(handles.size());
	for (unsigned i = 0; i < handles.size(); ++i)
	{
		if (handles[i] == 0)
		{
			free_vertices.push_back(i);
		}
	}

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
				Eigen::VectorXd projection = edge_vector - (dot_prod * normals.row(i).transpose());

				projections.push_back(projection);
			}
			// Find the longest projection
			Eigen::VectorXd lengths(projections.size());
			for (unsigned k = 0; k < projections.size(); ++k)
			{
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
	for (size_t i = 0; i < xis.size(); ++i)
	{
		xis_mat.row(i) = xis[i];
	}
	Eigen::MatrixXd free_n(free_vertices.size(), 3);
	for (size_t i = 0; i < free_vertices.size(); ++i)
	{
		free_n.row(i) = normals.row(free_vertices[i]);
	}

	Eigen::MatrixXd yis(free_vertices.size(), 3);
	for (size_t i = 0; i < free_vertices.size(); ++i)
	{
		Eigen::Vector3d free_n_vec = free_n.row(i);
		Eigen::Vector3d xis_vec = xis_mat.row(i);
		yis.row(i) = free_n_vec.cross(xis_vec);
	}

	Eigen::VectorXd dx = elementwise_dot_sum(displacements, xis_mat);
	Eigen::VectorXd dy = elementwise_dot_sum(displacements, yis);
	Eigen::VectorXd dn = elementwise_dot_sum(displacements, normals(free_vertices, Eigen::all));

	Eigen::MatrixXd displace_components(free_vertices.size(), 3);
	displace_components << dx, dy, dn;

	std::vector<Segment> segments(labels.maxCoeff() - 1);

	// Init the viewer
	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	viewer.plugins.push_back(&plugin);
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	plugin.widgets.push_back(&menu);

	Eigen::MatrixXd target_pos = V;

	// Add content to the default menu window
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("Mesh Deformation", ImGuiTreeNodeFlags_DefaultOpen))
		{
			static int segment = 1;
			bool updated = false;
			updated |= ImGui::SliderInt("Segment", &segment, 1, labels.maxCoeff());
			updated |= ImGui::SliderFloat("X", &segments[segment - 1].x, -1.0f, 1.0f);
			updated |= ImGui::SliderFloat("Y", &segments[segment - 1].y, -1.0f, 1.0f);
			updated |= ImGui::SliderFloat("Z", &segments[segment - 1].z, -1.0f, 1.0f);
			updated |= ImGui::SliderFloat("Alpha", &segments[segment - 1].alpha, -90.0f, 90.0f);
			updated |= ImGui::SliderFloat("Beta", &segments[segment - 1].beta, -90.0f, 90.0f);
			updated |= ImGui::SliderFloat("Gamma", &segments[segment - 1].gamma, -90.0f, 90.0f);

			if (updated)
			{
				move_handles(target_pos, V, labels, segment, segments[segment - 1].x, segments[segment - 1].y, segments[segment - 1].z, 
				segments[segment - 1].alpha, segments[segment - 1].beta, segments[segment - 1].gamma);
				position_deformer(target_pos, F, proj_i_indices_eigen, proj_j_indices_eigen, chol, A_fc, free_vertices, constrained_vertices, displace_components);
				viewer.data().clear();
				viewer.data().set_mesh(target_pos, F);
				viewer.data().set_colors(colors);
			}
		}
	};

	// Plot the mesh
	viewer.data().set_mesh(V, F);
	viewer.data().set_colors(colors);
	viewer.data().add_label(viewer.data().V.row(0) + viewer.data().V_normals.row(0).normalized() * 0.005, "Hello World!");
	viewer.launch();
}

Eigen::MatrixXd smooth_mesh(const Eigen::SparseMatrix<double> &A_fc, const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &A_free, const Eigen::Ref<const Eigen::MatrixXd> &vc)
{
	auto A_fcv_c = A_fc * vc;
	Eigen::MatrixXd v_free = A_free.solve(-A_fcv_c);
	return v_free;
}

Eigen::VectorXd elementwise_dot_sum(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
	return (a.array() * b.array()).rowwise().sum();
}

void move_handles(Eigen::MatrixXd &target_pos, const Eigen::MatrixXd &V, Eigen::ArrayXi &labels, int s, double x, double y, double z, double alpha, double beta, double gamma)
{
	Eigen::Array<bool, Eigen::Dynamic, 1> mask = (labels.array() == s);

	int count = mask.count();

	Eigen::MatrixXd v_slice(count, V.cols());
	int idx = 0;
	for (int i = 0; i < V.rows(); ++i)
	{
		if (mask(i))
		{
			v_slice.row(idx++) = V.row(i);
		}
	}

	v_slice.rowwise() += Eigen::RowVector3d(x, y, z);

	Eigen::Vector3d center = v_slice.colwise().mean();

	Eigen::Matrix3d rotation;
	rotation = Eigen::AngleAxisd(alpha * M_PI / 180.0, Eigen::Vector3d::UnitX()) *
						 Eigen::AngleAxisd(beta * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
						 Eigen::AngleAxisd(gamma * M_PI / 180.0, Eigen::Vector3d::UnitZ());

	v_slice = (v_slice.rowwise() - center.transpose()) * rotation.transpose();
	v_slice.rowwise() += center.transpose();

	idx = 0;
	for (int i = 0; i < V.rows(); ++i)
	{
		if (mask(i))
		{
			target_pos.row(i) = v_slice.row(idx++);
		}
	}
}

void position_deformer(Eigen::MatrixXd &target_pos, const Eigen::MatrixXi &F, Eigen::VectorXi &proj_i_indices, Eigen::VectorXi &proj_j_indices,
											 const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> &A_free, const Eigen::SparseMatrix<double> &A_fc,
											 std::vector<int> &free_vertices, std::vector<int> &constrained_vertices,
											 const Eigen::MatrixXd &displace_components)
{
	Eigen::MatrixXd target_free = smooth_mesh(A_fc, A_free, target_pos(constrained_vertices, Eigen::all));
	target_pos(free_vertices, Eigen::all) = target_free;

	Eigen::MatrixXd n_prime;
	igl::per_vertex_normals(target_pos, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, n_prime);

	Eigen::MatrixXd edge_vectors = target_pos(proj_j_indices, Eigen::all) - target_pos(proj_i_indices, Eigen::all);
	Eigen::MatrixXd normals = n_prime(proj_i_indices, Eigen::all);

	Eigen::VectorXd dot_products = (edge_vectors.array() * normals.array()).rowwise().sum();
	Eigen::MatrixXd normal_components = dot_products.replicate(1, 3).array() * normals.array();
	Eigen::MatrixXd projections = edge_vectors - normal_components;

	Eigen::VectorXd norms = projections.rowwise().norm();
	Eigen::MatrixXd xis_prime = projections.array().colwise() / norms.array();

	Eigen::MatrixXd yis_prime(free_vertices.size(), 3);
	for (size_t i = 0; i < free_vertices.size(); ++i)
	{
		Eigen::Vector3d free_n_vec = n_prime.row(free_vertices[i]);
		Eigen::Vector3d xis_vec = xis_prime.row(i);
		yis_prime.row(i) = free_n_vec.cross(xis_vec);
	}

	Eigen::MatrixXd d_transform(free_vertices.size(), 3);
	for (int i = 0; i < free_vertices.size(); ++i)
	{
		d_transform.row(i) = displace_components(i, 0) * xis_prime.row(i) +
												 displace_components(i, 1) * yis_prime.row(i) +
												 displace_components(i, 2) * n_prime.row(free_vertices[i]);
	}

	target_pos(free_vertices, Eigen::all) += d_transform;
}

void assign_colors(const Eigen::ArrayXi &labels, Eigen::MatrixXd &C)
{
	C.resize(labels.rows(), 3);
	Eigen::RowVector3d red(1.0, 0.0, 0.0);
	Eigen::RowVector3d green(0.0, 1.0, 0.0);
	Eigen::RowVector3d blue(0.0, 0.0, 1.0);
	Eigen::RowVector3d yellow(1.0, 1.0, 0.0);
	Eigen::RowVector3d gray(0.5, 0.5, 0.5);

	for (int i = 0; i < labels.rows(); ++i)
	{
		switch (labels(i))
		{
		case 0:
			C.row(i) = gray;
			break;
		case 1:
			C.row(i) = red;
			break;
		case 2:
			C.row(i) = green;
			break;
		case 3:
			C.row(i) = blue;
			break;
		case 4:
			C.row(i) = yellow;
		}
	}
}