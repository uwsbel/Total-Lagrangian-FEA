import sympy
import os

def save_to_latex_file(filename, sympy_object):
    """Converts a SymPy object to a LaTeX string and saves it to a file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to LaTeX string using 'plain' mode for compatibility.
    latex_string = sympy.latex(sympy_object, mode='plain')
    
    with open(filename, 'w') as f:
        f.write(latex_string)
    print(f"Saved {filename}")

def save_node_locations_to_latex(filename, node_coords):
    """Converts node coordinates to LaTeX format and saves to a file."""
    u, v, w, L, W = sympy.symbols('u v w L W')
    
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

def generate_shell_92_shape_functions():
    """
    Generates the shape functions for a 9-node shell element with 2 unknowns per node
    (r, r_w), using a monomial basis approach. This corresponds to the S3-92 element.
    The basis is a tensor product of (1, u, u^2), (1, v, v^2), and (1, w).
    """
    # 1. Define symbolic variables
    u, v, w = sympy.symbols('u v w')
    L, W = sympy.symbols('L W')

    # 2. Define the monomial basis from the tensor product of the specified polynomials.
    # This creates a systematic 18-term basis.
    b_u = [1, u, u**2]
    b_v = [1, v, v**2]
    
    # First, create the 9 terms for the 2D plane (u,v)
    b_2d_terms = [term_u * term_v for term_u in b_u for term_v in b_v]
    
    # Then, create the full 18-term basis by incorporating the (1, w) component
    b_1_part = sympy.Matrix(b_2d_terms)
    b_w_part = sympy.Matrix([w * term for term in b_2d_terms])
    
    basis_functions = sympy.Matrix([b_1_part, b_w_part])
    
    num_unknowns = len(basis_functions)
    
    # 3. Define the 9 nodal locations in the reference (u,v) plane at w=0
    # These form a 3x3 grid.
    node_coords = [
        {u: -L/2, v: -W/2, w: 0},  # Node 1 (bottom-left)
        {u:  L/2, v: -W/2, w: 0},  # Node 2 (bottom-right)
        {u:  L/2, v:  W/2, w: 0},  # Node 3 (top-right)
        {u: -L/2, v:  W/2, w: 0},  # Node 4 (top-left)
        {u:    0, v: -W/2, w: 0},  # Node 5 (bottom-mid)
        {u:  L/2, v:    0, w: 0},  # Node 6 (right-mid)
        {u:    0, v:  W/2, w: 0},  # Node 7 (top-mid)
        {u: -L/2, v:    0, w: 0},  # Node 8 (left-mid)
        {u:    0, v:    0, w: 0}   # Node 9 (center)
    ]
    num_nodes = len(node_coords)

    # 4. Establish the system of equations by evaluating the basis functions
    #    and their w-derivatives at the nine nodes.
    B_matrix_T = sympy.zeros(num_unknowns, num_unknowns)
    
    # Pre-calculate the derivative of the basis functions with respect to w
    basis_diff_w = sympy.diff(basis_functions, w)

    # Populate the B_matrix_T (transpose of the B matrix)
    for i in range(num_nodes):
        coords = node_coords[i]
        # There are 2 unknowns per node, so we fill 2 rows for each node.
        row_start_index = i * 2
        
        # Condition for the function value 'r'
        B_matrix_T[row_start_index, :] = basis_functions.subs(coords).T
        # Condition for the derivative 'r_w'
        B_matrix_T[row_start_index + 1, :] = basis_diff_w.subs(coords).T
        
    # The B matrix is the transpose of what we just built
    B_matrix = B_matrix_T.T
    
    # 5. Solve for the shape functions by inverting the B matrix.
    #    s(u,v,w) = B^(-1) * b(u,v,w)
    print("Inverting the 18x18 B matrix... this may take a moment.")
    B_matrix_inv = B_matrix.inv()
    print("Inversion complete.")
    
    shape_functions = B_matrix_inv * basis_functions

    return shape_functions, basis_functions, B_matrix_inv, B_matrix, node_coords

def verify_shell_92_shape_functions(shape_functions):
    """
    Verifies that the generated shape functions satisfy the Kronecker delta property
    at the nodal points and for the nodal w-derivatives.
    """
    # 1. Define symbolic variables
    u, v, w, L, W = sympy.symbols('u v w L W')
    
    # 2. Define nodal locations (must match generation)
    node_coords = [
        {u: -L/2, v: -W/2, w: 0}, {u:  L/2, v: -W/2, w: 0}, {u:  L/2, v:  W/2, w: 0},
        {u: -L/2, v:  W/2, w: 0}, {u:    0, v: -W/2, w: 0}, {u:  L/2, v:    0, w: 0},
        {u:    0, v:  W/2, w: 0}, {u: -L/2, v:    0, w: 0}, {u:    0, v:    0, w: 0}
    ]
    num_nodes = len(node_coords)
    num_shape_functions = len(shape_functions)
    
    # 3. Create the verification matrix
    verification_matrix = sympy.zeros(num_shape_functions, num_shape_functions)

    print("Verifying shape functions...")
    for j in range(num_shape_functions): # Iterate through columns (each shape function)
        s_j = shape_functions[j]
        s_j_dw = sympy.diff(s_j, w)

        for i in range(num_nodes): # Iterate through nodes
            coords = node_coords[i]
            row_start_index = i * 2
            
            # Check conditions for node i against shape function j
            verification_matrix[row_start_index,     j] = sympy.simplify(s_j.subs(coords))
            verification_matrix[row_start_index + 1, j] = sympy.simplify(s_j_dw.subs(coords))

    print("Verification complete.")
    return verification_matrix

if __name__ == '__main__':
    # --- Define Element Parameters ---
    element_type = "shell"
    num_nodes = 9
    num_unknowns_per_node = 2
    
    # --- Setup Output Directory ---
    folder_name = f"{element_type}_{num_nodes}{num_unknowns_per_node}"
    output_dir = os.path.join('latex_output', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")
    
    # --- Generation ---
    print("--- Generating Shape Functions for Shell S3-92 Element ---")
    shape_funcs, basis_funcs, B_inv_matrix, B_matrix, node_coords = generate_shell_92_shape_functions()

    # --- Verification ---
    print("\n--- Verifying Shape Functions ---")
    verification_mat = verify_shell_92_shape_functions(shape_funcs)
    is_identity = (verification_mat == sympy.eye(18))
    print(f"Verification successful: {is_identity}")

    # --- Saving Output for LaTeX ---
    print("\n--- Saving Generated LaTeX Code to Files ---")
    save_to_latex_file(os.path.join(output_dir, 'basis_functions.tex'), basis_funcs)
    save_to_latex_file(os.path.join(output_dir, 'B_matrix.tex'), B_matrix)
    save_to_latex_file(os.path.join(output_dir, 'B_inv_matrix.tex'), B_inv_matrix)
    save_to_latex_file(os.path.join(output_dir, 'shape_functions.tex'), shape_funcs)
    save_to_latex_file(os.path.join(output_dir, 'verification_matrix.tex'), verification_mat)
    save_node_locations_to_latex(os.path.join(output_dir, 'node_locations.tex'), node_coords)
    
    verification_status_file = os.path.join(output_dir, 'verification_status.txt')
    with open(verification_status_file, 'w') as f:
        f.write(f"Verification matrix is the identity matrix: {is_identity}\n")
    print(f"Saved {verification_status_file}")
    
    print(f"\nPipeline execution complete. Output saved to '{folder_name}/' folder.")
    print(f"You can now use the .tex files in your LaTeX report.")
