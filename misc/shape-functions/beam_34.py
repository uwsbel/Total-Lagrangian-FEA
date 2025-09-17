import sympy
import os

def save_to_latex_file(filename, sympy_object):
    """Converts a SymPy object to a LaTeX string and saves it to a file."""
    latex_string = sympy.latex(sympy_object, mode='plain')
    with open(filename, 'w') as f:
        f.write(latex_string)
    print(f"Saved {filename}")

def save_node_locations_to_latex(filename, node_coords):
    """Converts node coordinates to LaTeX format and saves to a file."""
    u, v, w, L = sympy.symbols('u v w L')
    
    latex_lines = []
    for i, coords in enumerate(node_coords, 1):
        u_val = coords[u]
        v_val = coords[v] 
        w_val = coords[w]
        latex_lines.append(f"\\text{{Node {i}}}: & \\quad ({sympy.latex(u_val)}, {sympy.latex(v_val)}, {sympy.latex(w_val)}) \\\\")
    
    latex_content = "\\begin{align*}\n" + "\n".join(latex_lines) + "\n\\end{align*}"
    
    with open(filename, 'w') as f:
        f.write(latex_content)
    print(f"Saved {filename}")

def generate_beam_34_shape_functions():
    """
    Generates the shape functions for a 3-node beam element with 4 unknowns per node
    (position and three gradients).
    """
    # 1. Define symbolic variables
    u, v, w = sympy.symbols('u v w')
    L = sympy.Symbol('L')
    
    # 2. Define the corrected monomial basis for 12 total unknowns.
    # Quintic in u, Quadratic in v and w components.
    basis_functions = sympy.Matrix([
        1, u, u**2, u**3, u**4, u**5,
        v, u*v, u**2*v,
        w, u*w, u**2*w
    ])
    num_unknowns = len(basis_functions)
    
    # 3. Define the nodal locations in standardized dictionary format
    node_coords = [
        {u: -L/2, v: 0, w: 0},  # Node 1
        {u:    0, v: 0, w: 0},  # Node 2
        {u:  L/2, v: 0, w: 0}   # Node 3
    ]

    # 4. Establish the system of equations. There are 12 nodal unknowns:
    # 4 at node 1, 4 at node 2, 4 at node 3
    B_T = sympy.zeros(num_unknowns, num_unknowns)
    basis_diff_u = sympy.diff(basis_functions, u)
    basis_diff_v = sympy.diff(basis_functions, v)
    basis_diff_w = sympy.diff(basis_functions, w)

    # Evaluate at node 1 (-L/2)
    B_T[0, :] = basis_functions.subs(node_coords[0]).T
    B_T[1, :] = basis_diff_u.subs(node_coords[0]).T
    B_T[2, :] = basis_diff_v.subs(node_coords[0]).T
    B_T[3, :] = basis_diff_w.subs(node_coords[0]).T
    
    # Evaluate at node 2 (0)
    B_T[4, :] = basis_functions.subs(node_coords[1]).T
    B_T[5, :] = basis_diff_u.subs(node_coords[1]).T
    B_T[6, :] = basis_diff_v.subs(node_coords[1]).T
    B_T[7, :] = basis_diff_w.subs(node_coords[1]).T

    # Evaluate at node 3 (L/2)
    B_T[8, :] = basis_functions.subs(node_coords[2]).T
    B_T[9, :] = basis_diff_u.subs(node_coords[2]).T
    B_T[10, :] = basis_diff_v.subs(node_coords[2]).T
    B_T[11, :] = basis_diff_w.subs(node_coords[2]).T
    
    # Display B matrix
    B = B_T.T
    print("Corrected B matrix for beam_34:")
    sympy.pprint(B)
    
    # 5. Solve for the shape functions
    B_inv = B.inv()
    shape_functions = B_inv * basis_functions

    return shape_functions, basis_functions, B_inv, node_coords

def verify_beam_34_shape_functions(shape_functions):
    """
    Verifies the generated shape functions for the beam_34 element.
    """
    u, v, w, L = sympy.symbols('u v w L')
    node_coords = [
        {u: -L/2, v: 0, w: 0},  # Node 1
        {u:    0, v: 0, w: 0},  # Node 2
        {u:  L/2, v: 0, w: 0}   # Node 3
    ]
    
    num_shape_functions = len(shape_functions)
    verification_matrix = sympy.zeros(num_shape_functions, num_shape_functions)

    for i, s_i in enumerate(shape_functions):
        s_i_du = sympy.diff(s_i, u)
        s_i_dv = sympy.diff(s_i, v)
        s_i_dw = sympy.diff(s_i, w)

        # Check at Node 1
        verification_matrix[0, i] = sympy.simplify(s_i.subs(node_coords[0]))
        verification_matrix[1, i] = sympy.simplify(s_i_du.subs(node_coords[0]))
        verification_matrix[2, i] = sympy.simplify(s_i_dv.subs(node_coords[0]))
        verification_matrix[3, i] = sympy.simplify(s_i_dw.subs(node_coords[0]))

        # Check at Node 2
        verification_matrix[4, i] = sympy.simplify(s_i.subs(node_coords[1]))
        verification_matrix[5, i] = sympy.simplify(s_i_du.subs(node_coords[1]))
        verification_matrix[6, i] = sympy.simplify(s_i_dv.subs(node_coords[1]))
        verification_matrix[7, i] = sympy.simplify(s_i_dw.subs(node_coords[1]))

        # Check at Node 3
        verification_matrix[8, i] = sympy.simplify(s_i.subs(node_coords[2]))
        verification_matrix[9, i] = sympy.simplify(s_i_du.subs(node_coords[2]))
        verification_matrix[10, i] = sympy.simplify(s_i_dv.subs(node_coords[2]))
        verification_matrix[11, i] = sympy.simplify(s_i_dw.subs(node_coords[2]))

    return verification_matrix

if __name__ == '__main__':
    # Define element parameters
    element_type = "beam"
    num_nodes = 3
    num_unknowns_per_node = 4
    
    # Create folder name based on element parameters
    folder_name = f"{element_type}_{num_nodes}{num_unknowns_per_node}"
    
    # Create the output directory
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")

    # --- Generation ---
    shape_funcs, basis_funcs, B_matrix_inv, node_coords = generate_beam_34_shape_functions()

    # --- Verification ---
    verification_mat = verify_beam_34_shape_functions(shape_funcs)
    is_identity = (verification_mat == sympy.eye(12))

    # --- Saving Output for LaTeX ---
    print("Saving generated LaTeX code to files...")
    save_to_latex_file(os.path.join(output_dir, 'basis_functions.tex'), basis_funcs)
    save_to_latex_file(os.path.join(output_dir, 'B_inv_matrix.tex'), B_matrix_inv)
    save_to_latex_file(os.path.join(output_dir, 'shape_functions.tex'), shape_funcs)
    save_to_latex_file(os.path.join(output_dir, 'verification_matrix.tex'), verification_mat)
    save_node_locations_to_latex(os.path.join(output_dir, 'node_locations.tex'), node_coords)
    
    with open(os.path.join(output_dir, 'verification_status.txt'), 'w') as f:
        f.write(str(is_identity))
    print(f"Saved {os.path.join(output_dir, 'verification_status.txt')}")
    
    print(f"\nPipeline execution complete. Output saved to the '{folder_name}/' folder.")
    print("You can now use a LaTeX template to compile a report from this output.")