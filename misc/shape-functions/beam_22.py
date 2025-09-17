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

def generate_beam_22_shape_functions():
    """
    Generates the shape functions for a 2-node beam element with 2 unknowns per node
    (position and u-gradient).
    """
    # 1. Define symbolic variables
    u, v, w = sympy.symbols('u v w')
    L = sympy.Symbol('L')
    
    # 2. Define the monomial basis for 4 total unknowns.
    # A standard choice for a Hermite beam is a cubic polynomial.
    basis_functions = sympy.Matrix([1, u, u**2, u**3])
    num_unknowns = len(basis_functions)
    
    # 3. Define the nodal locations in standardized dictionary format
    node_coords = [
        {u: -L/2, v: 0, w: 0},  # Node 1
        {u:  L/2, v: 0, w: 0}   # Node 2
    ]

    # 4. Establish the system of equations. The 4 nodal unknowns are:
    # x at node 1, dx/du at node 1
    # x at node 2, dx/du at node 2
    B12_T = sympy.zeros(num_unknowns, num_unknowns)
    basis_diff_u = sympy.diff(basis_functions, u)

    # Evaluate at node 1
    B12_T[0, :] = basis_functions.subs(node_coords[0]).T
    B12_T[1, :] = basis_diff_u.subs(node_coords[0]).T
    
    # Evaluate at node 2
    B12_T[2, :] = basis_functions.subs(node_coords[1]).T
    B12_T[3, :] = basis_diff_u.subs(node_coords[1]).T
    
    # Display B12 matrix
    B12 = B12_T.T
    print("B12 matrix for beam_22:")
    sympy.pprint(B12)
    
    # 5. Solve for the shape functions
    B12_inv = B12.inv()
    shape_functions = B12_inv * basis_functions

    return shape_functions, basis_functions, B12_inv, node_coords

def verify_beam_22_shape_functions(shape_functions):
    """
    Verifies the generated shape functions for the beam_22 element.
    """
    u, v, w, L = sympy.symbols('u v w L')
    node_coords = [
        {u: -L/2, v: 0, w: 0},  # Node 1
        {u:  L/2, v: 0, w: 0}   # Node 2
    ]
    
    num_shape_functions = len(shape_functions)
    verification_matrix = sympy.zeros(num_shape_functions, num_shape_functions)

    for i, s_i in enumerate(shape_functions):
        s_i_du = sympy.diff(s_i, u)

        # Check at Node 1
        verification_matrix[0, i] = sympy.simplify(s_i.subs(node_coords[0]))
        verification_matrix[1, i] = sympy.simplify(s_i_du.subs(node_coords[0]))

        # Check at Node 2
        verification_matrix[2, i] = sympy.simplify(s_i.subs(node_coords[1]))
        verification_matrix[3, i] = sympy.simplify(s_i_du.subs(node_coords[1]))

    return verification_matrix

if __name__ == '__main__':
    # Define element parameters
    element_type = "beam"
    num_nodes = 2
    num_unknowns_per_node = 2
    
    # Create folder name based on element parameters
    folder_name = f"{element_type}_{num_nodes}{num_unknowns_per_node}"
    
    # Create the output directory
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")

    # --- Generation ---
    shape_funcs, basis_funcs, B_matrix_inv, node_coords = generate_beam_22_shape_functions()

    # --- Verification ---
    verification_mat = verify_beam_22_shape_functions(shape_funcs)
    is_identity = (verification_mat == sympy.eye(4))

    # --- Saving Output for LaTeX ---
    print("\nSaving generated LaTeX code to files...")
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