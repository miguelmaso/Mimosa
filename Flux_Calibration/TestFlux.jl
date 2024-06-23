using Flux



x_train = LinRange(-1,1,10)


f_1(x) = x.^2;
f_2(x) = x.^3;
f_3(x) = x.^3 .+ 1.1;


y_1 = f_1(x_train)
y_2 = f_2(x_train)
y_3 = f_3(x_train)





model = Chain(
    Dense(1=>3, softplus),
    Dense(3=>3, softplus),
    Dense(3=>1, softplus),
)

opt = Flux.setup(Adam(0.05), model)
function loss(flux_model,x,y)
     ŷ = flux_model(x)
     Flux.mse(ŷ,y)
end;

function iterative_training(model, x_train, y_train)
    data = [(x_train,y_train)]
    epoch = 1
    iter = 1
    Losses = zeros(0)
    while  loss(model,x_train,y_train)>0.00005
     Flux.train!(loss, model, data, opt)
     L = loss(model, x_train, y_train)
     println("Epoch number $epoch with a loss $L ")
     iter += 1
     epoch += 1

      append!(Losses,L)
    end
    return Losses
end

Losses=iterative_training(model, x_train', y_3')

