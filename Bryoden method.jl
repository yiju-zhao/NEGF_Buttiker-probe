### Initialization of fermi level for Buttiker Probes
I_max = 1
N_iter = 1
Δμ = zeros(Ns)
I_BP = zeros(Ns)
J_inv = zeros(Ns,Ns)

### Searching loop of fermi level for Buttiker Probes
while (I_max > δI_tol)

    # Reset variables
    fill!(Iᴱ_BP,0.0)

    μ_BP += Δμ
    for i = 1:Ns
        if μ_BP[i] < μ₂[ind_Vd]
            μ_BP[i] = μ₂[ind_Vd]

        elseif μ_BP[i] > μ₁
            μ_BP[i] = μ₁
        end
    end
    println("Value of mu_BP is $μ_BP")

    if N_iter == 1

        NEGF_Results = pmap((Eᵢ,kyᵢ)-> NEGF_Map(Eᵢ,kyᵢ,Uᵢ,μ_BP,N_iter,ind_Vd),wp,E_map,ky_map,batch_size=nky)

        @inbounds for ii = 1:nE*nky

            Iᴱ_BP[Eky[ii,1],:] += NEGF_Results[ii][6]
            Jᴱ_BP[Eky[ii,1],:,:] += NEGF_Results[ii][7]
            NEGF_time[ii] = NEGF_Results[ii][8]
        end 

        I_sum = sum(Iᴱ_BP,dims=1)
        I_BP = I_sum[1,:]   
        I_max = maximum(abs.(I_BP))
        
        J_sum = sum(Jᴱ_BP,dims=1)
        J = J_sum[1,:,:]
        Δμ = -J\I_BP

        J_inv = inv(J)

        println("Value of I_max is $I_max")

    else
        
        NEGF_Results = pmap((Eᵢ,kyᵢ)-> NEGF_Map(Eᵢ,kyᵢ,Uᵢ,μ_BP,N_iter,ind_Vd),wp,E_map,ky_map,batch_size=nky)

        @inbounds for ii = 1:nE*nky

            Iᴱ_BP[Eky[ii,1],:] += NEGF_Results[ii][6]
            NEGF_time[ii] = NEGF_Results[ii][7]
        end

        I_sum = sum(Iᴱ_BP,dims=1)
        I_BP_new = I_sum[1,:]   
        I_max = maximum(abs.(I_BP_new))

        y = I_BP_new - I_BP
        J_inv = J_inv + (Δμ - J_inv*y)*Δμ'*J_inv/dot(Δμ, J_inv*y)

        # u = J_inv*I_BP
        # c = Δμ⋅(Δμ + u)
        # J_inv1 = J_inv - (kron(u,Δμ')*J_inv)/c
        # J_inv = J_inv1

        I_BP = I_BP_new
        Δμ = -J_inv*I_BP_new

        println("Value of I_max is $I_max")

    end

    N_iter += 1            

    if N_iter >= Nmax_I && I_max > δI_tol             # the upper limit for iteration number
        
        @inbounds for ii = 1:nE*nky
            Iᴱ[Eky[ii,1]] += NEGF_Results[ii][1]
            Tᴱ[Eky[ii,1]] += NEGF_Results[ii][2]
            LDOS[Eky[ii,1],:] += NEGF_Results[ii][3]
            nᴱ[Eky[ii,1],:,:] += NEGF_Results[ii][4]
            pᴱ[Eky[ii,1],:,:] += NEGF_Results[ii][5]
        end

        println("Error tolerance of I_max for Buttiker probe could not be achived in the given maximum number of iteration!") 
        
        break
    
    elseif I_max <= δI_tol
        @inbounds for ii = 1:nE*nky
            Iᴱ[Eky[ii,1]] += NEGF_Results[ii][1]
            Tᴱ[Eky[ii,1]] += NEGF_Results[ii][2]
            LDOS[Eky[ii,1],:] += NEGF_Results[ii][3]
            nᴱ[Eky[ii,1],:,:] += NEGF_Results[ii][4]
            pᴱ[Eky[ii,1],:,:] += NEGF_Results[ii][5]
        end

        println("Buttiker probe currents successfully converged!") 
    end
    
end

