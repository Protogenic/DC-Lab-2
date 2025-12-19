#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define G_CONST 6.67430e-11

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

std::vector<Particle> readParticles(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file");
    }
    
    int n;
    file >> n;
    std::vector<Particle> particles(n);
    
    for (int i = 0; i < n; i++) {
        file >> particles[i].x >> particles[i].y >> particles[i].z
             >> particles[i].vx >> particles[i].vy >> particles[i].vz;
        particles[i].mass = 1.0e24;
    }
    
    file.close();
    return particles;
}

void writeState(std::ofstream& out, double t, const std::vector<Particle>& particles) {
    out << t;
    for (const auto& p : particles) {
        out << " " << p.x << " " << p.y;
    }
    out << "\n";
}

__global__ void computeAccelerationsKernel(
    const double* x, const double* y, const double* z,
    const double* mass, double* ax, double* ay, double* az, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
    double xi = x[i], yi = y[i], zi = z[i];
    
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        
        double dx = x[j] - xi;
        double dy = y[j] - yi;
        double dz = z[j] - zi;
        double dist2 = dx*dx + dy*dy + dz*dz;
        double dist = sqrt(dist2);
        
        if (dist < 1e-10) continue;
        
        double force = G_CONST * mass[j] / (dist2 * dist);
        acc_x += force * dx;
        acc_y += force * dy;
        acc_z += force * dz;
    }
    
    ax[i] = acc_x;
    ay[i] = acc_y;
    az[i] = acc_z;
}

__global__ void updatePositionsKernel(
    double* x, double* y, double* z,
    double* vx, double* vy, double* vz,
    const double* ax, const double* ay, const double* az,
    double dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    x[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
    y[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
    z[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
    
    vx[i] += 0.5 * ax[i] * dt;
    vy[i] += 0.5 * ay[i] * dt;
    vz[i] += 0.5 * az[i] * dt;
}

__global__ void updateVelocitiesKernel(
    double* vx, double* vy, double* vz,
    const double* ax, const double* ay, const double* az,
    double dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    vx[i] += 0.5 * ax[i] * dt;
    vy[i] += 0.5 * ay[i] * dt;
    vz[i] += 0.5 * az[i] * dt;
}

class NBodyCUDA {
private:
    int n, num_blocks;
    double *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_mass, *d_ax, *d_ay, *d_az;
    
public:
    NBodyCUDA(const std::vector<Particle>& particles) {
        n = particles.size();
        num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vz, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mass, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ax, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ay, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_az, n * sizeof(double)));
        
        std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n), mass(n);
        
        for (int i = 0; i < n; i++) {
            x[i] = particles[i].x;
            y[i] = particles[i].y;
            z[i] = particles[i].z;
            vx[i] = particles[i].vx;
            vy[i] = particles[i].vy;
            vz[i] = particles[i].vz;
            mass[i] = particles[i].mass;
        }
        
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z, z.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, vx.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, vy.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, vz.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, mass.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    ~NBodyCUDA() {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_mass); cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    }
    
    void step(double dt) {
        computeAccelerationsKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_x, d_y, d_z, d_mass, d_ax, d_ay, d_az, n);
        CUDA_CHECK(cudaGetLastError());
        
        updatePositionsKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, dt, n);
        CUDA_CHECK(cudaGetLastError());
        
        computeAccelerationsKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_x, d_y, d_z, d_mass, d_ax, d_ay, d_az, n);
        CUDA_CHECK(cudaGetLastError());
        
        updateVelocitiesKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_vx, d_vy, d_vz, d_ax, d_ay, d_az, dt, n);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void getParticles(std::vector<Particle>& particles) {
        std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n);
        
        CUDA_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z.data(), d_z, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vx.data(), d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vy.data(), d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vz.data(), d_vz, n * sizeof(double), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < n; i++) {
            particles[i].x = x[i];
            particles[i].y = y[i];
            particles[i].z = z[i];
            particles[i].vx = vx[i];
            particles[i].vy = vy[i];
            particles[i].vz = vz[i];
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <tend> <input_file>" << std::endl;
        return 1;
    }
    
    double tend = std::atof(argv[1]);
    std::string filename = argv[2];
    
    std::vector<Particle> particles = readParticles(filename);
    int n = particles.size();
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    std::cout << "N-Body Simulation (CUDA)" << std::endl;
    std::cout << "Particles: " << n << std::endl;
    std::cout << "Device: " << deviceProp.name << std::endl;
    
    NBodyCUDA simulator(particles);
    
    double dt = 0.01;
    int num_steps = static_cast<int>(tend / dt);
    int output_every = std::max(1, num_steps / 1000);
    
    std::ofstream outfile("output.csv");
    outfile.precision(10);
    writeState(outfile, 0.0, particles);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int step = 1; step <= num_steps; step++) {
        simulator.step(dt);
        
        if (step % output_every == 0) {
            simulator.getParticles(particles);
            writeState(outfile, step * dt, particles);
        }
        
        if (step % (num_steps / 10) == 0) {
            std::cout << "Progress: " << (100.0 * step / num_steps) << "%" << std::endl;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    simulator.getParticles(particles);
    outfile.close();
    
    std::cout << "Time: " << (elapsed_ms / 1000.0) << " sec" << std::endl;
    std::cout << "Output: output.csv" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
