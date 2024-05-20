# Default target
all: clean param_visualization_n_dim_par2

# Target for building without OpenMP and pthread
param_visualization_n_dim_par2:
	g++ param_visualization_n_dim_par2.cpp -o param_visualization_n_dim_par2 -fopenmp -lpthread

# Target for building with OpenMP and pthread
single_core:
	g++ param_visualization_n_dim_par2.cpp -o param_visualization_n_dim_par2 

clean:
	rm -f param_visualization_n_dim_par2
