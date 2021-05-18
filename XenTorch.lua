--[[

    XenTorch NN Module created by JollyGameCrazy
    API: https://davoidd.github.io/XenTorch/
    v1.0.1

    WHAT'S NEW
    - Genetic Crossover

]]--

local XenTorch = {}

XenTorch.Special = {}

function XenTorch.Special.Sum(array_1, array_2)
    local array_new = array_1

    for i, layer in pairs(array_2) do
        if layer then
            for j, n in pairs(layer) do
                if type(n) == "table" then
                    for k, weight in pairs(n) do
                        array_new[i][j][k] += weight
                    end
                else
                    array_new[i][j] += n
                end
            end
        end
    end

    return array_new
end

function XenTorch.Special.Mean(array)
    local sum = 0

    for _, x in pairs(array) do
        sum += x
    end

    return sum / #array
end

function XenTorch.Special.Average(arrays, index)
    local sum = {}

    for i = 1, #arrays[1] do
        for _, array in pairs(arrays) do
            if not sum[i] then
                table.insert(sum, array[i])
            else
                sum[i] += array[i]
            end
        end
    end

    for k, v in pairs(sum) do
        sum[k] /= #arrays
    end

    return sum
end

XenTorch.Ops = {}

function XenTorch.Ops.Propagate(layer, input_array)
    local weighted_sum = {}
    for i = 1, #layer.weights[1], 1 do
        table.insert(weighted_sum, 0)
    end

    for n, neuron in pairs(layer.weights) do
        for w, weight in pairs(neuron) do
            weighted_sum[w] += input_array[n] * weight
        end
    end

    if layer.bias then
        for i, sum in pairs(weighted_sum) do
            weighted_sum[i] += layer.bias[i]
        end
    end

    return weighted_sum
end

function XenTorch.Ops.Mem_Propagate(model, input_array)
    local memory = {input_array}
    local passage = input_array

    for k, layer in pairs(model) do
        if layer["Activation"] then
            local layer_1 = layer["Activation"][1](passage)
            table.insert(memory, {layer_1, layer["Activation"][2](passage), "Activation"})
            passage = layer_1
        else
            local prop = XenTorch.Ops.Propagate(layer, passage)
            table.insert(memory, prop)
            passage = prop
        end
    end

    return memory
end

XenTorch.Ops.WD_2 = function() end

function XenTorch.Ops.WD_1(index_1, index_2, model, mem_prop, cost_f, y_label_array, real_output)
    local sum = 0

    for i, w in pairs(model[index_1].weights[index_2]) do
        local der_0 = 1
        local index_0t = index_1 <= #model and table.find(mem_prop[index_1 + 1], "Activation")

        if index_0t then
            der_0 = mem_prop[index_1 + 1][2][index_2]
        elseif index_1 <= #model then
            if table.find(mem_prop[index_1], "Activation") then
                der_0 = mem_prop[index_1][1][index_2]
            else
                der_0 = mem_prop[index_1][index_2]
            end
        end
        
        if index_1 < #model - 1 then
            if table.find(mem_prop[index_1 + 2], "Activation") then
                sum += der_0 * mem_prop[index_1 + 2][2][i] * XenTorch.Ops.WD_2(index_1 + 2, i, model, mem_prop, cost_f, y_label_array, real_output)
            elseif table.find(mem_prop[index_1 + 1], "Activation") then
                sum += der_0 * mem_prop[index_1 + 1][1][i] * XenTorch.Ops.WD_2(index_1 + 2, i, model, mem_prop, cost_f, y_label_array, real_output)
            else
                sum += der_0 * mem_prop[index_1 + 1][i] * XenTorch.Ops.WD_2(index_1 + 2, i, model, mem_prop, cost_f, y_label_array, real_output)
            end
        else
            if index_1 < #model then
                if table.find(mem_prop[index_1 + 2], "Activation") then
                    sum += der_0 * mem_prop[index_1 + 2][2][i] * cost_f[2](real_output, y_label_array)[i]
                elseif table.find(mem_prop[index_1 + 1], "Activation") then
                    sum += der_0 * mem_prop[index_1 + 1][1][i] * cost_f[2](real_output, y_label_array)[i]
                else
                    sum += der_0 * mem_prop[index_1 + 1][i] * cost_f[2](real_output, y_label_array)[i]
                end
            else
                sum += der_0 * cost_f[2](real_output, y_label_array)[i]
            end
        end
    end

    if sum == 0 then
        return 1
    else
        return sum / #model[index_1].weights[index_2]
    end
end

function XenTorch.Ops.WD_2(index_1, index_2, model, mem_prop, cost_f, y_label_array, real_output)
    local sum = 0

    for i, w in pairs(model[index_1].weights[index_2]) do
        local der_0 = 1
        local index_0t = index_1 <= #model and table.find(mem_prop[index_1 + 1], "Activation")

        if index_0t then
            der_0 = mem_prop[index_1 + 1][2][index_2]
        elseif index_1 <= #model then
            if table.find(mem_prop[index_1], "Activation") then
                der_0 = mem_prop[index_1][1][index_2]
            else
                der_0 = mem_prop[index_1][index_2]
            end
        end
        
        if index_1 < #model - 1 then
            if table.find(mem_prop[index_1 + 2], "Activation") then
                sum += der_0 * mem_prop[index_1 + 2][2][i] * XenTorch.Ops.WD_1(index_1 + 2, i, model, mem_prop, cost_f, y_label_array, real_output)
            elseif table.find(mem_prop[index_1 + 1], "Activation") then
                sum += der_0 * mem_prop[index_1 + 1][1][i] * XenTorch.Ops.WD_1(index_1 + 2, i, model, mem_prop, cost_f, y_label_array, real_output)
            else
                sum += der_0 * mem_prop[index_1 + 1][i] * XenTorch.Ops.WD_1(index_1 + 2, i, model, mem_prop, cost_f, y_label_array, real_output)
            end
        else
            if index_1 < #model then
                if table.find(mem_prop[index_1 + 2], "Activation") then
                    sum += der_0 * mem_prop[index_1 + 2][2][i] * cost_f[2](real_output, y_label_array)[i]
                elseif table.find(mem_prop[index_1 + 1], "Activation") then
                    sum += der_0 * mem_prop[index_1 + 1][1][i] * cost_f[2](real_output, y_label_array)[i]
                else
                    sum += der_0 * mem_prop[index_1 + 1][i] * cost_f[2](real_output, y_label_array)[i]
                end
            else
                sum += der_0 * cost_f[2](real_output, y_label_array)[i]
            end
        end
    end

    if sum == 0 then
        return 1
    else
        return sum / #model[index_1].weights[index_2]
    end
end

function XenTorch.Ops.WeightGradient(model, mem_prop, cost_f, y_label_array, real_output)
    local der_all = {}
    for k = 1, #model do
        if not model[k]["Activation"] then
            local d = k - 1
            local der_layer = {}

            if #model > 2 + d then
                for j, n in pairs(model[1 + d].weights) do
                    local der_neuron = {}

                    for i, w in pairs(n) do
                        if table.find(mem_prop[d + 1], "Activation") then
                            if table.find(mem_prop[d + 3], "Activation") then
                                table.insert(der_neuron, mem_prop[d + 1][1][j] * mem_prop[d + 3][2][i] * XenTorch.Ops.WD_1(3 + d, i, model, mem_prop, cost_f, y_label_array, real_output))
                            else
                                table.insert(der_neuron, mem_prop[d + 1][1][j] * mem_prop[d + 2][i] * XenTorch.Ops.WD_1(3 + d, i, model, mem_prop, cost_f, y_label_array, real_output))
                            end
                        else
                            if table.find(mem_prop[d + 3], "Activation") then
                                table.insert(der_neuron, mem_prop[d + 1][j] * mem_prop[d + 3][2][i] * XenTorch.Ops.WD_1(3 + d, i, model, mem_prop, cost_f, y_label_array, real_output))
                            else
                                table.insert(der_neuron, mem_prop[d + 1][j] * mem_prop[d + 2][i] * XenTorch.Ops.WD_1(3 + d, i, model, mem_prop, cost_f, y_label_array, real_output))
                            end
                        end
                    end

                    table.insert(der_layer, der_neuron)
                end
            else
                for j, n in pairs(model[1 + d].weights) do
                    local der_neuron = {}

                    for i, w in pairs(n) do
                        if table.find(mem_prop[d + 1], "Activation")  then
                            if table.find(mem_prop[d + 3], "Activation") then
                                table.insert(der_neuron, mem_prop[d + 1][1][j] * mem_prop[d + 3][2][i] * cost_f[2](real_output, y_label_array)[i])
                            else
                                table.insert(der_neuron, mem_prop[d + 1][1][j] * mem_prop[d + 2][i] * cost_f[2](real_output, y_label_array)[i])
                            end
                        elseif mem_prop[d + 3] then
                            if table.find(mem_prop[d + 3], "Activation") then
                                table.insert(der_neuron, mem_prop[d + 1][j] * mem_prop[d + 3][2][i] * cost_f[2](real_output, y_label_array)[i])
                            else
                                table.insert(der_neuron, mem_prop[d + 1][j] * mem_prop[d + 2][i] * cost_f[2](real_output, y_label_array)[i])
                            end
                        else
                            table.insert(der_neuron, mem_prop[d + 1][j] * cost_f[2](real_output, y_label_array)[i])
                        end
                    end

                    table.insert(der_layer, der_neuron)
                end
            end
            table.insert(der_all, der_layer)
        else
            table.insert(der_all, false)
        end
    end
    return der_all
end

function XenTorch.Ops.BiasGradient(model, mem_prop, cost_f, y_label_array, real_output)
    local der_all = {}
    for k = 1, #model do
        if not model[k]["Activation"] and type(model[k].bias) == "table" then
            local d = k - 1
            local der_layer = {}

            if #model > 2 + d then
                for i, b in pairs(model[1 + d].bias) do
                    local der_bias = 0

                    if table.find(mem_prop[d + 3], "Activation") then
                        der_bias = mem_prop[d + 3][2][i] * XenTorch.Ops.WD_1(3 + d, i, model, mem_prop, cost_f, y_label_array, real_output)
                    else
                        der_bias = mem_prop[d + 2][i] * XenTorch.Ops.WD_1(3 + d, i, model, mem_prop, cost_f, y_label_array, real_output)
                    end

                    table.insert(der_layer, der_bias)
                end
            else
                for i, b in pairs(model[1 + d].bias) do
                    local der_bias = 0

                    if mem_prop[d + 3] then
                        if table.find(mem_prop[d + 3], "Activation") then
                            der_bias = mem_prop[3 + d][2][i] * cost_f[2](real_output, y_label_array)[i]
                        else
                            der_bias = mem_prop[2 + d][i] * cost_f[2](real_output, y_label_array)[i]
                        end
                    else
                        der_bias = cost_f[2](real_output, y_label_array)[i]
                    end

                    table.insert(der_layer, der_bias)
                end
            end
            table.insert(der_all, der_layer)
        else
            table.insert(der_all, false)
        end
    end
    return der_all
end

XenTorch.Network = {}
XenTorch.Network.Model = {}
XenTorch.Network.Cost_f = {}

function XenTorch.Network.New(array, cost_f)
    new_class = {unpack(XenTorch.Network)}
    new_class.Model = array
    new_class.Cost_f = cost_f
    return new_class
end

function XenTorch.Network.Run(network, input_array)
    local passage = input_array
    for _, layer in pairs(network.Model) do
        if layer["Activation"] then
            passage = layer["Activation"][1](passage)
        else
            passage = XenTorch.Ops.Propagate(layer, passage)
        end
    end
    return passage
end

function XenTorch.Network.Error(network, x_set, y_set)
    local total_sum = 0
    local highest_error = 0

    for i, x_batch in pairs(x_set) do
        local sum = 0

        for j, x in pairs(x_batch) do
            local output = XenTorch.Network.Run(network, x)
            local y_label = y_set[i][j]
            local error = network.Cost_f[1](output, y_label)
            sum += XenTorch.Special.Mean(error)
        end
            
        sum /= #x_batch
        total_sum += sum

        if sum > highest_error then
            highest_error = sum
        end
    end

    return total_sum / #x_set, highest_error
end

function XenTorch.Network.BackPropagate(network, x_array, y_array, Optimizer, lr)
    local batch_size = #x_array

    if Optimizer == "SGD" then
        for a, x in pairs(x_array) do
            local y = y_array[a]
            local mem_prop = XenTorch.Ops.Mem_Propagate(network.Model, x)
            local real_output = XenTorch.Network.Run(network, x)

            for i, der_layer in pairs(XenTorch.Ops.WeightGradient(network.Model, mem_prop, network.Cost_f, y, real_output)) do
                if der_layer then
                    for j, der_neuron in pairs(der_layer) do
                        for k, der_weight in pairs(der_neuron) do
                            network.Model[i].weights[j][k] += -lr * der_weight
                        end
                    end
                end
            end

            for i, der_layer in pairs(XenTorch.Ops.BiasGradient(network.Model, mem_prop, network.Cost_f, y, real_output)) do
                if type(der_layer) == "table" then
                    for j, der_bias in pairs(der_layer) do
                        network.Model[i].bias[j] += -lr * der_bias
                    end
                end
            end
        end
    elseif Optimizer == "GD" then
        local weight_nudge = {}

        for a, x in pairs(x_array) do
            local y = y_array[a]
            local mem_prop = XenTorch.Ops.Mem_Propagate(network.Model, x)
            local real_output = XenTorch.Network.Run(network, x)
            if a ~= 1 then
                weight_nudge = XenTorch.Special.Sum(weight_nudge, XenTorch.Ops.WeightGradient(network.Model, mem_prop, network.Cost_f, y, real_output))
            else
                weight_nudge = XenTorch.Ops.WeightGradient(network.Model, mem_prop, network.Cost_f, y, real_output)
            end
        end

        for i, der_layer in pairs(weight_nudge) do
            if der_layer then
                for j, der_neuron in pairs(der_layer) do
                    for k, der_weight in pairs(der_neuron) do
                        network.Model[i].weights[j][k] += -lr * der_weight / batch_size
                    end
                end
            end
        end

        local bias_nudge = {}

        for a, x in pairs(x_array) do
            local y = y_array[a]
            local mem_prop = XenTorch.Ops.Mem_Propagate(network.Model, x)
            local real_output = XenTorch.Network.Run(network, x)
            if a ~= 1 then
                bias_nudge = XenTorch.Special.Sum(bias_nudge, XenTorch.Ops.BiasGradient(network.Model, mem_prop, network.Cost_f, y, real_output))
            else
                bias_nudge = XenTorch.Ops.BiasGradient(network.Model, mem_prop, network.Cost_f, y, real_output)
            end
        end

        for i, der_layer in pairs(weight_nudge) do
            if der_layer then
                for j, der_neuron in pairs(der_layer) do
                    if bias_nudge[i] and bias_nudge[i][j] then
                        network.Model[i].bias[j] += -lr * bias_nudge[i][j] / batch_size
                    end
                end
            end
        end
    else
        warn("Input Error: XenTorch.Network.BackPropagate(); invalid optimizer '" .. Optimizer .. "'")
        return nil
    end

    return network
end

function XenTorch.Network.FitData(network, x_train, y_train, Optimizer, lr, x_test, y_test, termination, epoch_num)
    local RunService = Game:GetService("RunService")
    local epochs = 0
    
    while RunService.Heartbeat:Wait() do
        print("_____EPOCH " .. epochs + 1 .. "_____")
        for i, x_batch in pairs(x_train) do
            network = XenTorch.Network.BackPropagate(network, x_batch, y_train[i], Optimizer, lr)
            epochs += 1
            --print(unpack(x_batch[1]))
            --print(unpack(XenTorch.Network.Run(network, x_batch[1])))
            RunService.Heartbeat:Wait()
        end
        
        local error, highest_error = XenTorch.Network.Error(network, x_test, y_test)

        print("_Error: " .. error .. "; Highest Error: " .. highest_error .. "_")

        if (epoch_num and epochs >= epoch_num) or (termination and error < termination) then
            break
        end
    end

    return network
end

XenTorch.nn = {}

function XenTorch.nn.Sequential(array, cost_f)
    return XenTorch.Network.New(array, cost_f)
end

function XenTorch.nn.Linear(input_dim, output_dim, bias)
    if bias == nil then
        bias = false
    end
    local frame = {weights = {}, bias = false}

    for m = 1, input_dim, 1 do
        local row = {}
        for n = 1, output_dim, 1 do
            table.insert(row, math.random())
        end
        table.insert(frame.weights, row)
    end

    if bias then
        local row = {}
        for n = 1, output_dim, 1 do
            if type(bias) == "boolean" then
                table.insert(row, 0)
            else
                table.insert(row, bias)
            end
        end
        frame.bias = row
    end

    return frame
end

function XenTorch.nn.Wise(func)
    local function new_func(array)
        output = {}
        for _, x in pairs(array) do
            table.insert(output, func(x))
        end
        return output
    end

    return new_func
end

function XenTorch.nn.Intellect(func)
    local function new_func(a_1, a_2)
        output = {}
        for i, x in pairs(a_1) do
            table.insert(output, func(x, a_2[i]))
        end
        return output
    end

    return new_func
end

function XenTorch.nn.ReLU(x)
    return math.max(0, x)
end

function XenTorch.nn.Sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

function XenTorch.nn.Softmax(array)
    local output = {}
    local sum = 0

    for _, v in pairs(array) do
        sum += math.exp(v)
    end

    for _, x in pairs(array) do
        table.insert(output, math.exp(x) / sum)
    end

    return output
end

XenTorch.nn.Cost = {}

function XenTorch.nn.Cost.MSE(y_hat, y)
    return math.pow((y_hat - y), 2)
end

XenTorch.nn.Prime = {}

function XenTorch.nn.Prime.ReLU(x)
    if x < 0 then
        return 0
    else
        return 1
    end
end

function XenTorch.nn.Prime.Sigmoid(x)
    local sigma = 1 / (1 + math.exp(-x))
    return sigma * (1 - sigma)
end

function XenTorch.nn.Prime.Softmax(array)
    local output = {}
    local smax_values = XenTorch.nn.Softmax(array)

    for i, x in pairs(array) do
        table.insert(output, smax_values[i] * (1 - smax_values[i]))

        --[[
        local derivatives = {}

        for j, v in pairs(array) do
            local derivative
            if i ~= j then
                derivative = -smax_values[i] * smax_values[j]
            else
                derivative = smax_values[i] * (1 - smax_values[i])
            end
            table.insert(derivatives, derivative)
        end
        table.insert(output, derivatives)
        ]]
    end

    return output
end

XenTorch.nn.Prime.Cost = {}

function XenTorch.nn.Prime.Cost.MSE(y_hat, y)
    return y_hat - y
end

XenTorch.nn.ReLU = XenTorch.nn.Wise(XenTorch.nn.ReLU)
XenTorch.nn.Sigmoid = XenTorch.nn.Wise(XenTorch.nn.Sigmoid)
XenTorch.nn.Cost.MSE = XenTorch.nn.Intellect(XenTorch.nn.Cost.MSE)

XenTorch.nn.Prime.ReLU = XenTorch.nn.Wise(XenTorch.nn.Prime.ReLU)
XenTorch.nn.Prime.Sigmoid = XenTorch.nn.Wise(XenTorch.nn.Prime.Sigmoid)
XenTorch.nn.Prime.Cost.MSE = XenTorch.nn.Intellect(XenTorch.nn.Prime.Cost.MSE)

XenTorch.Genetic = {}

XenTorch.Genetic = {}

function XenTorch.Genetic.Binary(Parents, Weight_Selectiveness, Bias_Selectiveness)
    local Catalogue = {}

    for _, parent in pairs(Parents) do
        for i, layer in pairs(parent.Model) do
            if not Catalogue[i] then
                table.insert(Catalogue, {layer})
            else
                table.insert(Catalogue[i], layer)
            end
        end
    end

    local offspring = XenTorch.nn.Sequential({}, Parents[1].Cost_f)

    if Weight_Selectiveness == "Layer" then
        for _, layer_catalogue in pairs(Catalogue) do
            table.insert(offspring.Model, layer_catalogue[math.random(1, #layer_catalogue)])
        end
    elseif Weight_Selectiveness == "Neuron" then
        for i, layer_catalogue in pairs(Catalogue) do
            if layer_catalogue[1].weights then
                table.insert(offspring.Model, layer_catalogue[1])
    
                for j, neuron in pairs(layer_catalogue[1].weights) do
                    offspring.Model[i].weights[j] = layer_catalogue[math.random(1, #layer_catalogue)].weights[j]
                end

                if Bias_Selectiveness == "Complete" and offspring[i].bias then
                    offspring.Model[i].bias = layer_catalogue[math.random(1, #layer_catalogue)].bias
                elseif Bias_Selectiveness == "Individual" and offspring.Model[i].bias then
                    for j, bias in pairs(offspring.Model[i].bias) do
                        offspring.Model.bias[j] = layer_catalogue[math.random(1, #layer_catalogue)].bias[j]
                    end
                end
            else
                table.insert(offspring.Model, layer_catalogue[math.random(1, #layer_catalogue)])
            end
        end
    elseif Weight_Selectiveness == "Weight" then
        for i, layer_catalogue in pairs(Catalogue) do
            if layer_catalogue[1].weights then
                table.insert(offspring.Model, layer_catalogue[1])
    
                for j, neuron in pairs(layer_catalogue[1].weights) do
                    for k, weight in pairs(neuron) do
                        offspring.Model[i].weights[j][k] = layer_catalogue[math.random(1, #layer_catalogue)].weights[j][k]
                    end
                end

                if Bias_Selectiveness == "Complete" and offspring.Model[i].bias then
                    offspring.Model[i].bias = layer_catalogue[math.random(1, #layer_catalogue)].bias
                elseif Bias_Selectiveness == "Individual" and offspring[i].bias then
                    for j, bias in pairs(offspring.Model[i].bias) do
                        offspring.Model.bias[j] = layer_catalogue[math.random(1, #layer_catalogue)].bias[j]
                    end
                end
            else
                table.insert(offspring, layer_catalogue[math.random(1, #layer_catalogue)])
            end
        end
    end

    return offspring
end

function XenTorch.Genetic.Mean(Parents, Activation)
    local offspring = XenTorch.nn.Sequential(Parents[1].Model, Parents[1].Cost_f)

    for i, layer in pairs(offspring.Model) do
        if layer.weights then
            for j, neuron in pairs(layer.weights) do
                local indexed_Parents = {}
                for _, parent in pairs(Parents) do
                    table.insert(indexed_Parents, parent.Model[i].weights[j])
                end

                offspring.Model[i].weights[j] = XenTorch.Special.Average(indexed_Parents)
            end

            if layer.bias then
                local indexed_Parents = {}
                for _, parent in pairs(Parents) do
                    table.insert(indexed_Parents, parent.Model[i].bias)
                end

                offspring.Model[i].bias = XenTorch.Special.Average(indexed_Parents)
            end
        elseif Activation then
            local raw_counts = {}

            for _, parent in pairs(Parents) do
                table.insert(raw_counts, parent.Model[i])
            end

            local highest_count = 0
            local highest_selections = {}

            for _, selection in pairs(raw_counts) do
                local count = 0

                repeat
                    table.remove(raw_counts, table.find(raw_counts, selection))
                    count += 1
                until table.find(raw_counts, selection) == nil

                if count > highest_count then
                    highest_count = count
                    highest_selections = {selection}
                elseif count == highest_count then
                    table.insert(highest_selections, selection)
                end
            end

            offspring.Model[i] = highest_selections[math.random(1, #highest_selections)]
        end
    end

    return offspring
end

function XenTorch.Genetic.Mutate(offspring, probability, m_factor)
    for i, layer in pairs(offspring.Model) do
        if layer.weights then
            if math.random() < probability[1] then
                for j, neuron in pairs(layer.weights) do
                    if math.random() < probability[2] then
                        for k, weight in pairs(neuron) do
                            if math.random() < probability[3] then
                                if math.random() > 0.5 then
                                    offspring.Model[i].weights[j][k] = weight * math.random() * m_factor
                                else
                                    offspring.Model[i].weights[j][k] = weight * -math.random() * m_factor
                                end
                            end
                        end
                    end
                end

                if layer.bias then
                    for j, bias in pairs(layer.bias) do
                        if math.random() < probability[4] then
                            if math.random() > 0.5 then
                                offspring.Model[i].bias[j] = bias * math.random() * m_factor
                            else
                                offspring.Model[i].bias[j] = bias * -math.random() * m_factor
                            end
                        end
                    end
                end
            end
        end
    end

    return offspring
end

XenTorch.Data = {}

function XenTorch.Data.Randomize(array_1, array_2)
    if array_2 == nil then
        local new_array = {}

        for m = 1, #array_1 do
            local i = math.random(1, #array_1)
            table.insert(new_array, array_1[i])
            table.remove(array_1, i)
        end

        return new_array
    else
        local new_array_1 = {}
        local new_array_2 = {}

        for m = 1, #array_1 do
            local i = math.random(1, #array_1)
            table.insert(new_array_1, array_1[i])
            table.insert(new_array_2, array_2[i])
            table.remove(array_1, i)
            table.remove(array_2, i)
        end

        return new_array_1, new_array_2
    end
end

function XenTorch.Data.Separate(x_labels, y_labels, batch_size, ordered, validation)
    if ordered == nil then
        ordered = false
    end

    if validation == nil then
        validation = false
    end

    if ordered == false then
        x_labels, y_labels = XenTorch.Data.Randomize(x_labels, y_labels)
    end

    local x_batches = {}
    local y_batches = {}

    local cache_x = {}
    local cache_y = {}

    for i = 1, #x_labels do
        if #cache_x < batch_size then
            table.insert(cache_x, x_labels[i])
            table.insert(cache_y, y_labels[i])
        end

        if #cache_x >= batch_size then
            table.insert(x_batches, cache_x)
            table.insert(y_batches, cache_y)
            cache_x, cache_y = {}, {}
        end
    end

    if validation == false then
        local train_set = {{}, {}}
        local test_set = {{}, {}}

        local index = math.floor(#x_batches * 0.75)

        for i = 1, #x_batches do
            if i <= index then
                table.insert(train_set[1], x_batches[i])
                table.insert(train_set[2], y_batches[i])
            else
                table.insert(test_set[1], x_batches[i])
                table.insert(test_set[2], y_batches[i])
            end
        end

        return train_set, test_set
    else
        local train_set = {{}, {}}
        local validation_set = {{}, {}}
        local test_set = {{}, {}}

        local index_1 = math.floor(#x_batches * 0.7)
        local index_2 = math.floor((x_batches - index_1) * 0.5)

        for i = 1, #x_batches do
            if i <= index_1 then
                table.insert(train_set[1], x_batches[i])
                table.insert(train_set[2], y_batches[i])
            elseif i <= index_2 then
                table.insert(validation_set[1], x_batches[i])
                table.insert(validation_set[2], y_batches[i])
            else
                table.insert(test_set[1], x_batches[i])
                table.insert(test_set[2], y_batches[i])
            end
        end

        return train_set, test_set, validation_set
    end
end

return XenTorch