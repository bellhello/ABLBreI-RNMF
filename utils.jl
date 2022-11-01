function soft_thresholding(A::AbstractArray{T}, s::Real) where T
    B = copy(A)
    n = length(B)
    @inbounds for i = 1 : n
        x = sign(B[i])*max(abs(B[i]) - s, 0)
        B[i] = x
    end
    return B
end