#' spinner
#'
#' @description Spinner is an implementation of Graph Nets based on torch. Graph Nets are a family of neural network architectures designed for processing graphs and other structured data. They consist of a set of message-passing operations, which propagate information between nodes and edges in the graph, and a set of update functions, which compute new node and edge features based on the received messages.
#'
#' @param graph A graph in igraph format (without name index for nodes).
#' @param target String. Predicted dimension. Options are: "node", "edge".
#' @param node_labels String. Character vector with labels of node features. In case of absent features, default to NA (automatic node embedding with selected method).
#' @param edge_labels String. Character vector with labels of edge features. In case of absent features, default to NA (automatic edge embedding with selected method).
#' @param context_labels String. Character vector with labels of context features. In case of absent features, default to NA (automatic context embedding with selected method).
#' @param direction String. Direction of message propagation. Options are: "from_head", "from_tail". Default to: "from_head".
#' @param sampling Positive numeric or integer. In case of huge graph, you can opt for a subgraph. Sampling dimension expressed in absolute value or percentage. Default: NA (no sampling).
#' @param threshold Numeric. Below this threshold (calculated on edge density), sampling is done on edges, otherwise on nodes. Default: 0.01.
#' @param method String. Embedding method in case of absent features. Options are: "null" (zeroed tensor), "laplacian", "adjacency". Default: "null".
#' @param node_embedding_size Integer. Size for node embedding. Default: 5.
#' @param edge_embedding_size Integer. Size for edge embedding. Default: 5.
#' @param context_embedding_size Integer. Size for node embedding. Default: 5.
#' @param update_order String. The order of message passing through nodes (n), edges (e) and context (c) for updating information. Available options are: "enc", "nec", "cen", "ecn", "nce", "cne". Default: "enc".
#' @param n_layers Integer. Number of graph net variant layers. Default: 1.
#' @param skip_shortcut Logical. Flag for applying skip shortcut after the graph net variant layers. Default: FALSE.
#' @param forward_layer Integer. Single integer vector with size for forward net layer. Default: 32 (layers with 32 nodes).
#' @param forward_activation String. Single character vector with activation for forward net layer. Available options are: "linear", "relu", "mish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "sigmoid", "tanh". Default: "relu".
#' @param forward_drop Numeric. Single numeric vector with drop out for forward net layer. Default: 0.3.
#' @param mode String. Aggregation method for message passing. Options are: "sum", "mean", "max". Default: "sum".
#' @param optimization String. Optimization method. Options are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam".
#' @param epochs Positive integer. Default: 100.
#' @param lr Positive numeric. Learning rate. Default: 0.01.
#' @param patience Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: 30.
#' @param weight_decay Positive numeric. L2-Regularization weight. Default: 0.001.
#' @param reps Positive integer. Number of repeated measures. Default: 1.
#' @param folds Positive integer. Number of folds for each repetition. Default: 3.
#' @param holdout Positive numeric. Percentage of nodes for testing (edges are computed accordingly). Default: 0.2.
#' @param verbose Logical. Default: TRUE
#' @param seed Random seed. Default: 42.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#'@return This function returns a list including:
#'\itemize{
#'\item graph: analyzed graph is returned (original graph or sampled subgraph).
#'\item model_description: general model description.
#'\item model_summary: summary for each torch module.
#'\item pred_fun: function to predict on new graph data (you need to add new nodes/edges to the original graph respecting the directionality).
#'\item cv_error: cross-validation error for each repetition and each fold. The error is a weighted normalized loss based on mse and binary cross-entropy (depending on the nature of each specific feature).
#'\item summary_errors: final summary of error during cross-validation and testing.
#'\item history: plot with loss for final training and testing.
#'\item time_log: computation time.
#' }
#'
#' @export
#'
#' @import torch
#' @importFrom purrr map map2 pmap keep discard map_lgl map_dbl map_if flatten map_dfr map2_dbl
#' @import ggplot2
#' @import tictoc
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period
#' @importFrom ggthemes theme_clean
#' @importFrom igraph get.vertex.attribute get.edge.attribute graph.attributes diameter remove.vertex.attribute vcount ecount as_edgelist line.graph subgraph set_vertex_attr subgraph.edges set_edge_attr random.graph.game embed_adjacency_matrix embed_laplacian_matrix edge_density degree difference is.igraph add.vertices add.edges
#' @import abind
#' @importFrom rlist list.insert
#' @importFrom fastDummies dummy_columns
#' @importFrom entropy entropy
#' @importFrom utils tail
#' @importFrom stats loess predict quantile sd
#'
#'
spinner <- function(graph, target, node_labels = NA, edge_labels = NA, context_labels = NA, direction = "from_head",
                        sampling = NA, threshold = 0.01, method = "null", node_embedding_size = 5, edge_embedding_size = 5, context_embedding_size = 5,
                        update_order = "enc", n_layers = 3, skip_shortcut = FALSE, forward_layer = 32, forward_activation = "relu", forward_drop = 0.3, mode = "sum",
                        optimization = "adam", epochs = 100, lr = 0.01, patience = 30, weight_decay = 0.001,
                        reps = 1, folds = 3, holdout = 0.2, verbose = TRUE, seed = 42)
{
  tic.clearlog()
  tic("time")

  ####if(cuda_is_available()){dev <- "cuda"} else {dev <- "cpu"}###
  dev <- if (
    identical(Sys.getenv("NOT_CRAN"), "true") &&
    .Platform$OS.type != "windows" &&
    nzchar(system.file(package = "torch")) &&
    isTRUE(tryCatch(asNamespace("torch")$cuda_is_available(), error = function(e) FALSE))
  ) "cuda" else "cpu"

  ###INPUT CHECK
  target <- match.arg(arg = target, choices = c("node", "edge"), several.ok = FALSE)
  direction <- match.arg(arg = direction, choices = c("from_head", "from_tail"), several.ok = FALSE)
  method <- match.arg(arg = method, choices = c("null", "laplacian", "adjacency"), several.ok = FALSE)
  forward_activation <- match.arg(arg = forward_activation, choices = c("linear", "relu", "mish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "sigmoid", "tanh"), several.ok = TRUE)
  mode <- match.arg(arg = mode, choices = c("sum", "max", "mean"), several.ok = FALSE)
  optimization <- match.arg(arg = optimization, choices = c("adam", "adagrad", "adadelta", "sgd", "asgd", "rprop", "rmsprop"), several.ok = FALSE)
  update_order <- match.arg(arg = update_order, choices = c("enc", "nec", "cen", "ecn", "nce", "cne"), several.ok = FALSE)

  if(length(forward_layer) > 1){forward_layer <- forward_layer[1]}
  if(length(forward_activation) > 1){forward_activation <- forward_activation[1]}
  if(length(forward_drop) > 1){forward_drop <- forward_drop[1]}

  ####PREPARATION
  #if(is.directed(graph) & direction == "undirected"){graph <- as.undirected(graph, mode = "collapse", edge.attr.comb = "random")}
  #if(!is.directed(graph) & direction != "undirected"){graph <- as.directed(graph, mode = "mutual")}
  if(!is.igraph(graph)){stop("graph required in igraph format")}
  if(!is.null(get.vertex.attribute(graph, "name"))){graph <- remove.vertex.attribute(graph, "name")}
  if(is.numeric(sampling) & sampling > 0 & (sampling < vcount(graph) | sampling < ecount(graph))){graph <- graph_sampling(graph, samp = sampling, threshold, seed)}

  n_nodes <- vcount(graph)
  n_edges <- ecount(graph)

  if(node_embedding_size < 2){node_embedding_size <- 2; message("setting node embedding size to minimum (2)")}
  node_schema <- feature_analyzer(graph, labels = node_labels, type = "node", method, node_embedding_size, mode)
  node_features <- node_schema$features
  node_dims <- dim(node_features)
  node_input_dim <- node_dims[2]
  node_idx <- 1:n_nodes

  node_norm_index <- unlist(map2(node_schema$group_init[node_schema$num_types], node_schema$group_end[node_schema$num_types], ~ .x:.y))
  node_norm_model <- normalizer(node_features, node_norm_index)
  node_features <- node_norm_model$norm_features

  if(edge_embedding_size < 2){edge_embedding_size <- 2; message("setting edge embedding size to minimum (2)")}
  edge_schema <- feature_analyzer(graph, labels = edge_labels, type = "edge", method, edge_embedding_size, mode)
  edge_features <- edge_schema$features
  edge_dims <- dim(edge_features)
  edge_input_dim <- edge_dims[2]
  edge_idx <- 1:edge_dims[1]

  edge_norm_index <- unlist(map2(edge_schema$group_init[edge_schema$num_types], edge_schema$group_end[edge_schema$num_types], ~ .x:.y))
  edge_norm_model <- normalizer(edge_features, edge_norm_index)
  edge_features <- edge_norm_model$norm_features

  if(context_embedding_size < 2){context_embedding_size <- 2; message("setting context embedding size to minimum (2)")}
  context_schema <- feature_analyzer(graph, labels = context_labels, type = "context", method, context_embedding_size, mode)
  context_features <- context_schema$features
  context_dims <- dim(context_features)
  context_input_dim <- context_dims[2]
  context_idx <- 1

  edge_list <- as_edgelist(graph)

  target_schema <- switch(target, "node" = node_schema, "edge" = edge_schema, "context" = context_schema)
  target_norm <- switch(target, "node" = node_norm_model, "edge" = edge_norm_model)

  node_feat_names <- colnames(node_features)
  edge_feat_names <- colnames(edge_features)
  context_feat_names <- colnames(context_features)

  #######TRAIN, VALIDATION & TEST
  node_train_idx <- sample(node_idx, round(n_nodes * (1-holdout)), replace = FALSE)
  node_train_length <- length(node_train_idx)
  node_train <- node_features[node_train_idx,, drop=FALSE]
  edge_train_index <- (edge_list[, 1] %in% node_train_idx) & (edge_list[, 2] %in% node_train_idx)

  edgelist_train <- edge_list[edge_train_index,, drop = FALSE]
  edge_train <- edge_features[edge_train_index,, drop = FALSE]

  node_test_idx <- setdiff(node_idx, node_train_idx)
  node_test_length <- length(node_test_idx)
  node_test <- node_features[node_test_idx,, drop=FALSE]
  edge_test_index <- (edge_list[, 1] %in% node_test_idx) & (edge_list[, 2] %in% node_test_idx)
  edgelist_test <- edge_list[edge_test_index,, drop = FALSE]
  edge_test <- edge_features[edge_test_index,, drop = FALSE]

  set.seed(seed)
  cv_index <- replicate(reps, sample(folds, node_train_length, replace = TRUE), simplify = FALSE)
  cv_index <- flatten(purrr::map(cv_index, ~ mapply(function(fold) .x == fold, fold = 1:folds, SIMPLIFY = FALSE)))
  train_idx <- map(cv_index, ~ node_train_idx[!.x])
  val_idx <- map(cv_index, ~ node_train_idx[.x])

  torch_manual_seed(seed)
  cv_models <- replicate(reps * folds, nn_graph_model(target, mode, n_layers, node_input_dim, edge_input_dim, context_input_dim, forward_layer, forward_activation, forward_drop, target_schema, dev), simplify = FALSE)
  cross_validation <- pmap(list(cv_models, train_idx, val_idx), ~ training_function(model = ..1, direction, node_features, edge_features, edge_list, ..2, ..3, context_features, target, target_schema, optimization, weight_decay, lr, epochs, patience, verbose, dev, update_order, skip_shortcut))

  final <- nn_graph_model(target, mode, n_layers, node_input_dim, edge_input_dim, context_input_dim, forward_layer, forward_activation, forward_drop, target_schema, dev)
  final_training <- training_function(model = final, direction, node_features, edge_features, edge_list, node_train_idx, node_test_idx, context_features, target, target_schema, optimization, weight_decay, lr, epochs, patience, verbose, dev, update_order, skip_shortcut)

  train_errors <- purrr::map(cross_validation, ~ matrix(tail(.x$train_history, 1), 1, 1))
  train_errors <- Reduce(rbind, train_errors)

  val_errors <- purrr::map(cross_validation, ~  matrix(tail(.x$val_history, 1), 1, 1))
  val_errors <- Reduce(rbind, val_errors)

  cv_errors <- cbind(expand.grid(folds = 1:folds, reps = 1:reps), train = train_errors, validation = val_errors)
  cv_errors <- cv_errors[, c(2, 1, 3, 4)]

  model <- final_training$model
  train_errors <- final_training$train_history
  test_errors <- final_training$val_history
  summary_errors <- c(train = tail(train_errors, 1), validation = colMeans(cv_errors[, 4, drop = FALSE]), test = tail(test_errors, 1))

  train_history <- final_training$train_history
  val_history <- final_training$val_history

  ###TRAINING PLOT
  act_epochs <- length(train_history)
  x_ref_point <- c(quantile(1:act_epochs, 0.15), quantile(1:act_epochs, 0.75))
  y_ref_point <- c(quantile(val_history, 0.75), quantile(train_history, 0.15))

  train_data <- data.frame(epochs = 1:act_epochs, train_loss = train_history)

  history <- ggplot(train_data) +
    geom_point(aes(x = epochs, y = train_loss), col = "blue", shape = 1, size = 1) +
    geom_smooth(col="darkblue", aes(x = epochs, y = train_loss), se=FALSE, method = "loess")

  val_data <- data.frame(epochs = 1:act_epochs, val_loss = val_history)

  history <- history + geom_point(aes(x = epochs, y = val_loss), val_data, col = "orange", shape = 1, size = 1) +
    geom_smooth(aes(x = epochs, y = val_loss), val_data, col="darkorange", se=FALSE, method = "loess")

  history <- history + ylab("Loss") + xlab("Epochs") +
    annotate("text", x = x_ref_point[1], y = y_ref_point[1], label = "TESTING SET", col = "darkorange", hjust = 0, vjust= 0) + ###SINCE THIS IS THE FINAL SET, WE ARE TESTING
    annotate("text", x = x_ref_point[2], y = y_ref_point[2], label = "TRAINING SET", col = "darkblue", hjust = 0, vjust= 0) +
    ggthemes::theme_clean() +
    ylab("weighted normalized loss")

  ###PREDICTION
  pred_fun <- function(new_graph)
  {
    #if(is.directed(new_graph) & direction == "undirected"){new_graph <- as.undirected(new_graph, mode = "collapse", edge.attr.comb = "random")}
    #if(!is.directed(new_graph) & direction != "undirected"){new_graph <- as.directed(new_graph, mode = "mutual")}
    if(!is.igraph(new_graph)){stop("graph required in igraph format")}

    feat_names <- target_schema$feat_names
    group_init <- target_schema$group_init
    group_end <- target_schema$group_end
    target_names <- map2(group_init, group_end, ~ feat_names[.x:.y])
    num_types <- target_schema$num_types
    feat_sizes <- target_schema$feat_sizes

    if(!is.null(get.vertex.attribute(new_graph, "name"))){new_graph <- remove.vertex.attribute(new_graph, "name")}

    delta_graph <- difference(new_graph, graph)
    previous_count <- vcount(graph)
    updated_count <- vcount(new_graph)
    n_new_nodes <- updated_count - previous_count
    n_new_edges <- ecount(delta_graph)

    same_node_attributes <- !is.null(node_labels) && all(node_labels %in% names(get.vertex.attribute(delta_graph)))
    same_edge_attributes <- !is.null(edge_labels) && all(edge_labels %in% names(get.edge.attribute(delta_graph)))
    same_context_attributes <- !is.null(context_labels) && all(context_labels %in% names(graph.attributes(delta_graph)))

    if(n_new_nodes > 0){
      if(same_node_attributes){selected_node_labels <- node_labels} else {selected_node_labels <- NA}
      delta_node_schema <- feature_analyzer(delta_graph, labels = selected_node_labels, type = "node", method, node_input_dim, mode)

      ###NORMALIZATION
      node_means <- node_norm_model$num_means
      node_scales <- node_norm_model$num_scales
      new_node_features <- mapply(function(i) (delta_node_schema$features[,i] - node_means[i])/node_scales[i], i = 1:ncol(delta_node_schema$features))

      new_node_features <- tail(new_node_features, n_new_nodes)###BECAUSE OF IGRAPH ODD WAY OF MANAGING DIFFERENCE OPERATION AT NODE LEVEL
      updated_node_features <- smart_rbind(node_features, new_node_features)
    }
    else {updated_node_features <- node_features}

    if(n_new_edges > 0){
      if(same_edge_attributes){selected_edge_labels <- edge_labels} else {selected_edge_labels <- NA}
      delta_edge_schema <- feature_analyzer(delta_graph, labels = selected_edge_labels, type = "edge", method, edge_input_dim, mode)
      new_edge_features <- delta_edge_schema$features

      ###NORMALIZATION
      edge_means <- edge_norm_model$num_means
      edge_scales <- edge_norm_model$num_scales
      new_edge_features <- mapply(function(i) (delta_edge_schema$features[,i] - edge_means[i])/edge_scales[i], i = 1:ncol(delta_edge_schema$features))

      updated_edge_features <- smart_rbind(edge_features, new_edge_features)
    }
    else {updated_edge_features <- edge_features}

    if(n_new_edges > 0){new_edgelist <- as_edgelist(delta_graph)}
    if(n_new_edges > 0){updated_edgelist <- rbind(edge_list, new_edgelist)} else {updated_edgelist <- edge_list}

    if(same_context_attributes){selected_context_labels <- context_labels} else {selected_context_labels <- NA}
    delta_context_schema <- feature_analyzer(delta_graph, labels = selected_context_labels, type = "context", method, context_input_dim, mode)
    updated_context_features <- delta_context_schema$features

    dir_list <- switch(direction, "undirected" = list(updated_edgelist[,1], updated_edgelist[,2]), "from_head" = list(updated_edgelist[,1], NULL), "from_tail" = list(NULL, updated_edgelist[,2]))
    pred <- model(updated_node_features, updated_edge_features, updated_context_features, from_node = dir_list[[1]], to_node = dir_list[[2]], update_order, skip_shortcut)
    if((target == "node" & n_new_nodes > 0) | (target == "edge" & n_new_edges > 0)){pred <- map(pred, ~ tail(.x, switch(target, "node" = n_new_nodes, "edge" = n_new_edges)))}

    ###RESCALING
    if(any(num_types) && !is.na(target_norm$num_scales))
    {
      num_means <- target_norm$num_means
      num_scales <- target_norm$num_scales

      if(length(feat_sizes) > 1)
      {
      group_index <- 1:length(feat_sizes[num_types])
      group_means <- split(num_means, unlist(map2(group_index, feat_sizes[num_types], ~ rep(.x, each = .y))))
      group_scales <- split(num_scales, unlist(map2(group_index, feat_sizes[num_types], ~ rep(.x, each = .y))))
      num_pred <- pmap(list(pred[num_types], group_means, group_scales), ~ invert_scale(..1, ..2, ..3, dev))
      pred[num_types] <- num_pred
      pred <- unlist(pred)###NEED TO FLATTEN THE INTERMEDIATE LIST LAYER INTRODUCED BY PMAP, BUT FLATTEN DON'T WORK WITH TORCH TENSORS
      }

      if(length(feat_sizes)==1)
      {
      pred <- invert_scale(pred, num_means, num_scales, dev)
      pred <- list(pred)
      }
    }

    pred <- map2(pred, target_names, ~ {x <- as.matrix(.x$to(device = "cpu")); colnames(x) <- .y; return(x)})

    if(target == "node" & n_new_nodes > 0){pred_idx <-(previous_count + 1):updated_count}
    if(target == "node" & n_new_nodes == 0){pred_idx <- 1:previous_count}
    if(target == "edge" & n_new_edges > 0){pred_idx <- new_edgelist; colnames(pred_idx) <- c("from", "to")}
    if(target == "edge" & n_new_edges == 0){pred_idx <- edge_list; colnames(pred_idx) <- c("from", "to")}

    if(target == "node"){pred <- map(pred, ~ cbind(nodes = pred_idx, .x))}
    if(target == "edge"){pred <- map(pred, ~ cbind(pred_idx, .x))}
    ###if(target == "context"){pred <- map(pred, ~ .x)}

    names(pred) <- switch(target, "node" = node_labels, "edge" = edge_labels, "context" = context_labels)

    return(pred)
  }


  modules <- c(paste0("GraphNetLayer", 1:n_layers), paste0("classif", 1:sum(target_schema$fct_types)), paste0("regr", 1:sum(target_schema$num_types)))
  model_summary <- map(modules, ~ model$modules[[.x]])
  names(model_summary) <- modules
  model_summary <- discard(model_summary, is.null)

  tot_parameters <- sum(map_dbl(flatten(map(model_summary, ~ .x$parameters)), ~ prod(dim(.x))))

  model_description <- paste0("model with ", n_layers, " GraphNet layers, ", sum(target_schema$fct_types)," classification tasks and ", sum(target_schema$num_types)," regression tasks (", tot_parameters," parameters)")

  toc(log = TRUE)
  time_log <- seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  ###OUTCOMES
  outcome <- list(graph = graph, model_description = model_description, model_summary = model_summary, pred_fun = pred_fun, cv_errors = cv_errors, summary_errors = summary_errors, history = history, time_log = time_log)

  return(outcome)
}



###SUPPORT FUNCTIONS FOR SPINNER

globalVariables(c("train_loss", "val_loss"))

###
nn_mish <- nn_module(
  "nn_mish",
  initialize = function() {self$softplus <- nn_softplus()},
  forward = function(x) {x * torch_tanh(self$softplus(x))})

###
nn_bent <- nn_module(
  "nn_bent",
  initialize = function() {},
  forward = function(x) {(torch_sqrt(x^2 + 1) - 1)/2 + x})

###
nn_activ <- nn_module(
  "nn_activ",
  initialize = function(act, dim = 2)
  {
    if(act == "linear"){self$activ <- nn_identity()}
    if(act == "mish"){self$activ <- nn_mish()}
    if(act == "relu"){self$activ <- nn_relu()}
    if(act == "leaky_relu"){self$activ <- nn_leaky_relu()}
    if(act == "celu"){self$activ <- nn_celu()}
    if(act == "elu"){self$activ <- nn_elu()}
    if(act == "gelu"){self$activ <- nn_gelu()}
    if(act == "selu"){self$activ <- nn_selu()}
    if(act == "bent"){self$activ <- nn_bent()}
    if(act == "softmax"){self$activ <- nn_softmax(dim)}
    if(act == "softmin"){self$activ <- nn_softmin(dim)}
    if(act == "softsign"){self$activ <- nn_softsign()}
    if(act == "sigmoid"){self$activ <- nn_sigmoid()}
    if(act == "tanh"){self$activ <- nn_tanh()}
  },
  forward = function(x)
  {
    x <- self$activ(x)
  })


###
nn_dense_network <- nn_module(
  "nn_dense_network",
  initialize = function(form, activ, drop, dev)
  {
    sequence <- 1:(length(form) - 1)
    self$sequence <- nn_buffer(sequence)

    layers <- paste0("layer", sequence)
    self$layers <- nn_buffer(layers)
    map2(layers, sequence, ~ {self[[.x]] <- nn_linear(in_features = form[.y], out_features = form[.y + 1], bias = TRUE)$to(device = dev)})

    activations <- paste0("activation", sequence)
    self$activations <- nn_buffer(activations)
    map2(activations, sequence, ~ {self[[.x]] <- nn_activ(act = activ[.y])})

    dropouts <- paste0("dropout", sequence)
    self$dropouts <- nn_buffer(dropouts)
    map2(dropouts, sequence, ~ {self[[.x]] <- nn_dropout(p = drop[.y])})

    self$norm <- nn_batch_norm1d(num_features = form[1])$to(device = dev)
  },

  forward = function(x)
  {
    sequence <- self$sequence
    layers <- self$layers
    activations <- self$activations
    dropouts <- self$dropouts

    x <- self$norm(x)

    for(i in sequence)
    {
      lyr <- layers[i]
      act <- activations[i]
      drp <- dropouts[i]
      x <- self[[lyr]](x)
      x <-  self[[act]](x)
      x <- self[[drp]](x)
    }

    x <- self$norm(x)

    return(x)
  }
)

###
nn_pooling_from_edges_to_nodes_layer <- nn_module(
  "nn_pooling_from_edges_to_nodes_layer",
  initialize = function(edge_input_dim, node_input_dim, dev)
  {
    self$dev <- nn_buffer(dev)
    self$linear_transf <- nn_linear(edge_input_dim, node_input_dim)$to(device = dev)
  },

  forward = function(edge_features, node_features, from_node, to_node, mode)
  {
    dev <- self$dev
    transformed_edge_features <- self$linear_transf(edge_features)

    if(!is.null(from_node)){
      split_by_from_node <-  mapply(function(i) transformed_edge_features[from_node == i,], i = unique(from_node))
      lone_nodes_from <- setdiff(1:nrow(node_features), unique(from_node))
      if(length(lone_nodes_from)>0) {for(p in lone_nodes_from){split_by_from_node <- list.insert(split_by_from_node, p, torch_zeros(1, ncol(node_features), dtype = torch_float(), device = dev))}}
    }

    if(!is.null(to_node)){
      split_by_to_node <-  mapply(function(i) transformed_edge_features[to_node == i,], i = unique(to_node))
      lone_nodes_to <- setdiff(1:nrow(node_features), unique(to_node))
      if(length(lone_nodes_to)>0) {for(p in lone_nodes_to){split_by_to_node <- list.insert(split_by_to_node, p, torch_zeros(1, ncol(node_features), dtype = torch_float(), device = dev))}}
    }

    if(mode == "sum"){
      if(!is.null(from_node)){
        stacked_from <- torch_stack(map(split_by_from_node, ~ torch_sum(.x, dim = 1, keepdim = TRUE)), dim = 1)
        pooled_edges_from <- torch_transpose(stacked_from, 2, 3)$squeeze(dim = 3)}
      else {pooled_edges_from <- torch_zeros_like(node_features, dtype = torch_float(), device = dev)}

      if(!is.null(to_node)){
        stacked_to <- torch_stack(map(split_by_to_node, ~ torch_sum(.x, dim = 1, keepdim = TRUE)), dim = 1)
        pooled_edges_to <- torch_transpose(stacked_to, 2, 3)$squeeze(dim = 3)}
      else {pooled_edges_to <- torch_zeros_like(node_features, dtype = torch_float(), device = dev)}

      updated_nodes <- pooled_edges_from + pooled_edges_to + node_features}

    if(mode == "max"){
      if(!is.null(from_node)){
        stacked_from <- torch_stack(map(split_by_from_node, ~ torch_max(.x, dim = 1, keepdim = TRUE)[[1]]), dim = 1)
        pooled_edges_from <- torch_transpose(stacked_from, 2, 3)$squeeze(dim = 3)}
      else {pooled_edges_from <- torch_zeros_like(node_features, dtype = torch_float(), device = dev)}

      if(!is.null(to_node)){
        stacked_to <- torch_stack(map(split_by_to_node, ~ torch_max(.x, dim = 1, keepdim = TRUE)[[1]]), dim = 1)
        pooled_edges_to <- torch_transpose(stacked_to, 2, 3)$squeeze(dim = 3)}
      else {pooled_edges_to <- torch_zeros_like(node_features, dtype = torch_float(), device = dev)}

      pooled_list <- mapply(function(i) torch_max(torch_stack(list(pooled_edges_from[i,], pooled_edges_to[i,], node_features[i,]), dim = 1), dim = 1, keepdim = TRUE)[[1]], i = 1:nrow(node_features))
      updated_nodes <- torch_stack(pooled_list, dim = 2)$squeeze(dim=1)}


    if(mode == "mean"){
      if(!is.null(from_node)){
        stacked_from <- torch_stack(map(split_by_from_node, ~ torch_mean(.x, dim = 1, keepdim = TRUE)), dim = 1)
        pooled_edges_from <- torch_transpose(stacked_from, 2, 3)$squeeze(dim = 3)}
      else {pooled_edges_from <- torch_zeros_like(node_features, dtype = torch_float(), device = dev)}

      if(!is.null(to_node)){
        stacked_to <- torch_stack(map(split_by_to_node, ~ torch_mean(.x, dim = 1, keepdim = TRUE)), dim = 1)
        pooled_edges_to <- torch_transpose(stacked_to, 2, 3)$squeeze(dim = 3)}
      else {pooled_edges_to <- torch_zeros_like(node_features, dtype = torch_float(), device = dev)}

      pooled_list <- mapply(function(i) torch_mean(torch_stack(list(pooled_edges_from[i,], pooled_edges_to[i,], node_features[i,]), dim = 1), dim = 1, keepdim = TRUE), i = 1:nrow(node_features))
      updated_nodes <- torch_stack(pooled_list, dim = 2)$squeeze(dim=1)}

    return(updated_nodes)
  })

###
nn_pooling_from_nodes_to_edges_layer <- nn_module(
  "nn_pooling_from_nodes_to_edges_layer",
  initialize = function(node_input_dim, edge_input_dim, dev)
  {
    self$dev <- nn_buffer(dev)
    self$linear_transf <- nn_linear(node_input_dim, edge_input_dim)$to(device = dev)
  },

  forward = function(node_features, edge_features, from_node, to_node, mode)
  {
    dev <- self$dev
    transformed_node_features <- self$linear_transf(node_features)

    if(!is.null(from_node)){pooled_nodes_from <-  torch_stack(mapply(function(i) transformed_node_features[i,], i = from_node), dim = 1)} else {pooled_nodes_from <- torch_zeros_like(edge_features, dtype = torch_float(), device = dev)}
    if(!is.null(to_node)){pooled_nodes_to <-  torch_stack(mapply(function(i) transformed_node_features[i,], i = to_node), dim = 1)} else {pooled_nodes_to <- torch_zeros_like(edge_features, dtype = torch_float(), device = dev)}

    if(mode == "sum"){updated_edges <- pooled_nodes_from + pooled_nodes_to + edge_features}
    if(mode == "max"){updated_edges <- torch_max(torch_stack(list(pooled_nodes_from, pooled_nodes_to, edge_features), dim = 3), dim = -1)[[1]]}
    if(mode == "mean"){updated_edges <- torch_mean(torch_stack(list(pooled_nodes_from, pooled_nodes_to, edge_features), dim = 3), dim = -1)}

    return(updated_edges)
  })

###
nn_pooling_from_context_to_edges_layer <- nn_module(
  "nn_pooling_from_context_to_edges_layer",
  initialize = function(context_input_dim, edge_input_dim, dev)
  {
    self$linear_transf <- nn_linear(context_input_dim, edge_input_dim)$to(device = dev)
  },

  forward = function(context_features, edge_features, mode)
  {
    transformed_context_features <- self$linear_transf(context_features)
    pooled_context <-  torch_stack(replicate(nrow(edge_features), transformed_context_features), dim = 2)$squeeze(dim = 1)

    if(mode == "sum"){updated_edges <- pooled_context + edge_features}
    if(mode == "max"){updated_edges <- torch_max(torch_stack(list(pooled_context, edge_features), dim = 3), dim = -1)[[1]]}
    if(mode == "mean"){updated_edges <- torch_mean(torch_stack(list(pooled_context, edge_features), dim = 3), dim = -1)}

    return(updated_edges)
  })

###
nn_pooling_from_context_to_nodes_layer <- nn_module(
  "nn_pooling_from_context_to_nodes_layer",
  initialize = function(context_input_dim, node_input_dim, dev)
  {
    self$linear_transf <- nn_linear(context_input_dim, node_input_dim)$to(device = dev)
  },

  forward = function(context_features, node_features, mode)
  {
    transformed_context_features <- self$linear_transf(context_features)
    pooled_context <-  torch_stack(replicate(nrow(node_features), transformed_context_features), dim = 2)$squeeze(dim = 1)

    if(mode == "sum"){updated_nodes <- pooled_context + node_features}
    if(mode == "max"){updated_nodes <- torch_max(torch_stack(list(pooled_context, node_features), dim = 3), dim = -1)[[1]]}
    if(mode == "mean"){updated_nodes <- torch_mean(torch_stack(list(pooled_context, node_features), dim = 3), dim = -1)}

    return(updated_nodes)
  })

###
nn_pooling_from_nodes_to_context_layer <- nn_module(
  "nn_pooling_from_nodes_to_context_layer",
  initialize = function(node_input_dim, context_input_dim, dev)
  {
    self$linear_transf <- nn_linear(node_input_dim, context_input_dim)$to(device = dev)
  },

  forward = function(node_features, context_features, mode)
  {
    transformed_node_features <- self$linear_transf(node_features)

    if(mode == "sum"){pooled_context <- torch_sum(transformed_node_features, 1, keepdim = TRUE)}
    if(mode == "max"){pooled_context <- torch_max(transformed_node_features, 1, keepdim = TRUE)[[1]]}
    if(mode == "mean"){pooled_context <- torch_mean(transformed_node_features, 1, keepdim = TRUE)}

    concat_tensor <- torch_cat(list(pooled_context, context_features), dim = 1)

    if(mode == "sum"){updated_context <- torch_sum(concat_tensor, 1, keepdim = TRUE)}
    if(mode == "max"){updated_context <- torch_max(concat_tensor, 1, keepdim = TRUE)[[1]]}
    if(mode == "mean"){updated_context <- torch_mean(concat_tensor, 1, keepdim = TRUE)}

    return(updated_context)
  })

###
nn_pooling_from_edges_to_context_layer <- nn_module(
  "nn_pooling_from_edges_to_context_layer",
  initialize = function(edge_input_dim, context_input_dim, dev)
  {
    self$linear_transf <- nn_linear(edge_input_dim, context_input_dim)$to(device = dev)
  },

  forward = function(edge_features, context_features, mode)
  {
    transformed_edge_features <- self$linear_transf(edge_features)

    if(mode == "sum"){pooled_context <- torch_sum(transformed_edge_features, 1, keepdim = TRUE)}
    if(mode == "max"){pooled_context <- torch_max(transformed_edge_features, 1, keepdim = TRUE)[[1]]}
    if(mode == "mean"){pooled_context <- torch_mean(transformed_edge_features, 1, keepdim = TRUE)}

    concat_tensor <- torch_cat(list(pooled_context, context_features), dim = 1)

    if(mode == "sum"){updated_context <- torch_sum(concat_tensor, 1, keepdim = TRUE)}
    if(mode == "max"){updated_context <- torch_max(concat_tensor, 1, keepdim = TRUE)[[1]]}
    if(mode == "mean"){updated_context <- torch_mean(concat_tensor, 1, keepdim = TRUE)}

    return(updated_context)
  })

###
nn_graph_net_layer<- nn_module(
  "nn_graph_net_layer",
  initialize = function(node_input_dim, edge_input_dim, context_input_dim, dnn_form, dnn_activ, dnn_drop, dev)
  {
    self$context_to_edge <- nn_pooling_from_context_to_edges_layer(context_input_dim, edge_input_dim, dev)
    self$context_to_node <- nn_pooling_from_context_to_nodes_layer(context_input_dim, node_input_dim, dev)
    self$edge_to_context <- nn_pooling_from_edges_to_context_layer(edge_input_dim, context_input_dim, dev)
    self$edge_to_node <- nn_pooling_from_edges_to_nodes_layer(edge_input_dim, node_input_dim, dev)
    self$node_to_context <- nn_pooling_from_nodes_to_context_layer(node_input_dim, context_input_dim, dev)
    self$node_to_edge <- nn_pooling_from_nodes_to_edges_layer(node_input_dim, edge_input_dim, dev)
    self$node_fusion <- nn_linear(2, 1)$to(device = dev)
    self$edge_fusion <- nn_linear(2, 1)$to(device = dev)
    self$context_fusion <- nn_linear(2, 1)$to(device = dev)
    self$independent_layer <- nn_graph_independent_forward_layer(node_input_dim, edge_input_dim, context_input_dim, dnn_form, dnn_activ, dnn_drop, dev)
  },

  forward = function(node_features, edge_features, context_features, from_node, to_node, mode, update_order)
  {
    if(update_order == "enc"){
      updated_edges_part_one <- self$context_to_edge(context_features, edge_features, mode)
      updated_edges_part_two <- self$node_to_edge(node_features, edge_features, from_node, to_node, mode)
      updated_edges <- self$edge_fusion(torch_stack(list(updated_edges_part_one, updated_edges_part_two), dim = 3))
      updated_edges <- updated_edges$squeeze(dim = 3)

      updated_nodes_part_one <- self$context_to_node(context_features, node_features, mode)
      updated_nodes_part_two <- self$edge_to_node(updated_edges, node_features, from_node, to_node, mode)
      updated_nodes <- self$node_fusion(torch_stack(list(updated_nodes_part_one, updated_nodes_part_two), dim = 3))
      updated_nodes <- updated_nodes$squeeze(dim = 3)

      updated_context_part_one <- self$edge_to_context(updated_edges, context_features, mode)
      updated_context_part_two <- self$node_to_context(updated_nodes, context_features, mode)
      updated_context <- self$context_fusion(torch_stack(list(updated_context_part_one, updated_context_part_two), dim = 3))
      updated_context <- updated_context$squeeze(dim = 3)
    }

    if(update_order == "nec"){
      updated_nodes_part_one <- self$context_to_node(context_features, node_features, mode)
      updated_nodes_part_two <- self$edge_to_node(edge_features, node_features, from_node, to_node, mode)
      updated_nodes <- self$node_fusion(torch_stack(list(updated_nodes_part_one, updated_nodes_part_two), dim = 3))
      updated_nodes <- updated_nodes$squeeze(dim = 3)

      updated_edges_part_one <- self$context_to_edge(context_features, edge_features, mode)
      updated_edges_part_two <- self$node_to_edge(updated_nodes, edge_features, from_node, to_node, mode)
      updated_edges <- self$edge_fusion(torch_stack(list(updated_edges_part_one, updated_edges_part_two), dim = 3))
      updated_edges <- updated_edges$squeeze(dim = 3)

      updated_context_part_one <- self$edge_to_context(updated_edges, context_features, mode)
      updated_context_part_two <- self$node_to_context(updated_nodes, context_features, mode)
      updated_context <- self$context_fusion(torch_stack(list(updated_context_part_one, updated_context_part_two), dim = 3))
      updated_context <- updated_context$squeeze(dim = 3)
    }

    if(update_order == "cne"){
      updated_context_part_one <- self$edge_to_context(edge_features, context_features, mode)
      updated_context_part_two <- self$node_to_context(node_features, context_features, mode)
      updated_context <- self$context_fusion(torch_stack(list(updated_context_part_one, updated_context_part_two), dim = 3))
      updated_context <- updated_context$squeeze(dim = 3)

      updated_nodes_part_one <- self$context_to_node(updated_context, node_features, mode)
      updated_nodes_part_two <- self$edge_to_node(edge_features, node_features, from_node, to_node, mode)
      updated_nodes <- self$node_fusion(torch_stack(list(updated_nodes_part_one, updated_nodes_part_two), dim = 3))
      updated_nodes <- updated_nodes$squeeze(dim = 3)

      updated_edges_part_one <- self$context_to_edge(updated_context, edge_features, mode)
      updated_edges_part_two <- self$node_to_edge(updated_nodes, edge_features, from_node, to_node, mode)
      updated_edges <- self$edge_fusion(torch_stack(list(updated_edges_part_one, updated_edges_part_two), dim = 3))
      updated_edges <- updated_edges$squeeze(dim = 3)
    }

    if(update_order == "cen"){
      updated_context_part_one <- self$edge_to_context(edge_features, context_features, mode)
      updated_context_part_two <- self$node_to_context(node_features, context_features, mode)
      updated_context <- self$context_fusion(torch_stack(list(updated_context_part_one, updated_context_part_two), dim = 3))
      updated_context <- updated_context$squeeze(dim = 3)

      updated_edges_part_one <- self$context_to_edge(updated_context, edge_features, mode)
      updated_edges_part_two <- self$node_to_edge(node_features, edge_features, from_node, to_node, mode)
      updated_edges <- self$edge_fusion(torch_stack(list(updated_edges_part_one, updated_edges_part_two), dim = 3))
      updated_edges <- updated_edges$squeeze(dim = 3)

      updated_nodes_part_one <- self$context_to_node(updated_context, node_features, mode)
      updated_nodes_part_two <- self$edge_to_node(updated_edges, node_features, from_node, to_node, mode)
      updated_nodes <- self$node_fusion(torch_stack(list(updated_nodes_part_one, updated_nodes_part_two), dim = 3))
      updated_nodes <- updated_nodes$squeeze(dim = 3)
    }

    if(update_order == "ecn"){
      updated_edges_part_one <- self$context_to_edge(context_features, edge_features, mode)
      updated_edges_part_two <- self$node_to_edge(node_features, edge_features, from_node, to_node, mode)
      updated_edges <- self$edge_fusion(torch_stack(list(updated_edges_part_one, updated_edges_part_two), dim = 3))
      updated_edges <- updated_edges$squeeze(dim = 3)

      updated_context_part_one <- self$edge_to_context(updated_edges, context_features, mode)
      updated_context_part_two <- self$node_to_context(node_features, context_features, mode)
      updated_context <- self$context_fusion(torch_stack(list(updated_context_part_one, updated_context_part_two), dim = 3))
      updated_context <- updated_context$squeeze(dim = 3)

      updated_nodes_part_one <- self$context_to_node(updated_context, node_features, mode)
      updated_nodes_part_two <- self$edge_to_node(updated_edges, node_features, from_node, to_node, mode)
      updated_nodes <- self$node_fusion(torch_stack(list(updated_nodes_part_one, updated_nodes_part_two), dim = 3))
      updated_nodes <- updated_nodes$squeeze(dim = 3)
    }

    if(update_order == "nce"){
      updated_nodes_part_one <- self$context_to_node(context_features, node_features, mode)
      updated_nodes_part_two <- self$edge_to_node(edge_features, node_features, from_node, to_node, mode)
      updated_nodes <- self$node_fusion(torch_stack(list(updated_nodes_part_one, updated_nodes_part_two), dim = 3))
      updated_nodes <- updated_nodes$squeeze(dim = 3)

      updated_context_part_one <- self$edge_to_context(edge_features, context_features, mode)
      updated_context_part_two <- self$node_to_context(updated_nodes, context_features, mode)
      updated_context <- self$context_fusion(torch_stack(list(updated_context_part_one, updated_context_part_two), dim = 3))
      updated_context <- updated_context$squeeze(dim = 3)

      updated_edges_part_one <- self$context_to_edge(updated_context, edge_features, mode)
      updated_edges_part_two <- self$node_to_edge(updated_nodes, edge_features, from_node, to_node, mode)
      updated_edges <- self$edge_fusion(torch_stack(list(updated_edges_part_one, updated_edges_part_two), dim = 3))
      updated_edges <- updated_edges$squeeze(dim = 3)
    }

    out <- self$independent_layer(updated_nodes, updated_edges, updated_context)

    return(out)
  })

###
nn_graph_independent_forward_layer <- nn_module(
  "nn_graph_independent_forward_layer",
  initialize = function(node_input_dim, edge_input_dim, context_input_dim, dnn_form, dnn_activ, dnn_drop, dev)
  {
    if(!is.null(node_input_dim))
    {
      node_dnn_form <- c(node_input_dim, dnn_form, node_input_dim)
      node_nlayers <- length(node_dnn_form) - 1
      if(length(dnn_activ) != node_nlayers){node_dnn_activ <- rep(dnn_activ[1], node_nlayers)} else {node_dnn_activ <- dnn_activ}
      if(length(dnn_drop) != node_nlayers){node_dnn_drop <- rep(dnn_drop[1], node_nlayers)} else {node_dnn_drop <- dnn_drop}
      self$node_dnn <- nn_dense_network(node_dnn_form, node_dnn_activ, node_dnn_drop, dev)###FORM MUST INCLUDE INPUT AND OUTPUT FEATS, ACTIV E DROP LENGTH LIKE FORM - 1
    }

    if(!is.null(edge_input_dim))
    {
      edge_dnn_form <- c(edge_input_dim, dnn_form, edge_input_dim)
      edge_nlayers <- length(edge_dnn_form) - 1
      if(length(dnn_activ) != edge_nlayers){edge_dnn_activ <- rep(dnn_activ[1], edge_nlayers)} else {edge_dnn_activ <- dnn_activ}
      if(length(dnn_drop) != edge_nlayers){edge_dnn_drop <- rep(dnn_drop[1], edge_nlayers)} else {edge_dnn_drop <- dnn_drop}
      self$edge_dnn <- nn_dense_network(edge_dnn_form, edge_dnn_activ, edge_dnn_drop, dev)
    }

    if(!is.null(context_input_dim))
    {
      context_dnn_form <- c(context_input_dim, dnn_form, context_input_dim)
      context_nlayers <- length(context_dnn_form) - 1
      if(length(dnn_activ) != context_nlayers){context_dnn_activ <- rep(dnn_activ[1], context_nlayers)} else {context_dnn_activ <- dnn_activ}
      if(length(dnn_drop) != context_nlayers){context_dnn_drop <- rep(dnn_drop[1], context_nlayers)} else {context_dnn_drop <- dnn_drop}
      self$context_dnn <- nn_dense_network(context_dnn_form, context_dnn_activ, context_dnn_drop, dev)
    }
  },
  forward = function(node_features, edge_features, context_features)
  {
    if(!is.null(node_features)){node_features <- self$node_dnn(node_features)} else {node_features <- NULL}
    if(!is.null(edge_features)){edge_features <- self$edge_dnn(edge_features)} else {edge_features <- NULL}
    if(!is.null(context_features)){context_features <- self$context_dnn(context_features)} else {context_features <- NULL}

    out <- list(node_features = node_features, edge_features = edge_features, context_features = context_features)
    return(out)
  })

###
nn_graph_model <- nn_module(
  "nn_graph_model",
  initialize = function(target, mode, n_layers, node_input_dim, edge_input_dim, context_input_dim, dnn_form, dnn_activ, dnn_drop, target_schema, dev)
  {
    self$dev <- nn_buffer(dev)
    self$target <- nn_buffer(target)
    self$mode <- nn_buffer(mode)

    layers <- paste0("GraphNetLayer", 1:n_layers)
    self$layers <- nn_buffer(layers)

    map(layers, ~ {self[[.x]] <- nn_graph_net_layer(node_input_dim, edge_input_dim, context_input_dim, dnn_form, dnn_activ, dnn_drop, dev)})

    group_init <- target_schema$group_init
    group_end <- target_schema$group_end
    num_types <- target_schema$num_types
    fct_types <- target_schema$fct_types

    self$group_init <- nn_buffer(group_init)
    self$group_end <- nn_buffer(group_end)
    self$num_types <- nn_buffer(num_types)
    self$fct_types <- nn_buffer(fct_types)

    if(any(fct_types))
    {
      group_sizes <- group_end[fct_types] - group_init[fct_types] + 1
      n_groups <- length(group_sizes)
      classif_names <- paste0("classif", 1:n_groups)
      self$classif_names <- nn_buffer(classif_names)
      for(i in 1:n_groups){if(group_sizes[i] > 1){self[[classif_names[i]]] <- nn_softmax(dim = 2)} else {self[[classif_names[i]]] <- nn_sigmoid()}}
    }

    if(any(num_types))
    {
      group_sizes <- group_end[num_types] - group_init[num_types] + 1
      n_groups <- length(group_sizes)
      regr_names <- paste0("regr", 1:n_groups)
      self$regr_names <- nn_buffer(regr_names)
      for(i in 1:n_groups){self[[regr_names[i]]] <- nn_linear(group_sizes[i], group_sizes[i])$to(device = dev)}
    }

  },

  forward = function(node_features, edge_features, context_features, from_node = NULL, to_node = NULL, update_order, skip_shortcut)
  {
    dev <- self$dev
    target <- self$target
    mode <- self$mode

    node_features <- torch_tensor(node_features, dtype = torch_float(), device = dev)
    edge_features <- torch_tensor(edge_features, dtype = torch_float(), device = dev)
    context_features <- torch_tensor(context_features, dtype = torch_float(), device = dev)

    orig_node_features <- node_features
    orig_edge_features <- edge_features
    orig_context_features <- context_features

    layers <- self$layers

    for(i in 1:length(layers))
    {
      lyr <- layers[i]
      interim <- self[[lyr]](node_features, edge_features, context_features, from_node, to_node, mode, update_order)
      node_features <- interim$node_features
      edge_features <- interim$edge_features
      context_features <- interim$context_features
    }

    if(skip_shortcut == TRUE)
    {
      node_features <- node_features + orig_node_features
      edge_features <- edge_features + orig_edge_features
      context_features <- context_features + orig_context_features
    }

    group_init <- self$group_init
    group_end <- self$group_end
    num_types <- self$num_types
    fct_types <- self$fct_types
    classif_names <- self$classif_names
    regr_names <- self$regr_names

    classif_outcomes <- list()
    regr_outcomes <- list()

    if(target == "node")
    {
      if(!is.null(classif_names)){classif_outcomes <- pmap(list(classif_names, group_init[fct_types], group_end[fct_types]), ~ self[[..1]](node_features[, ..2:..3, drop = FALSE]))}
      if(!is.null(regr_names)){regr_outcomes <- pmap(list(regr_names, group_init[num_types], group_end[num_types]), ~ self[[..1]](node_features[, ..2:..3, drop = FALSE]))}
    }

    if(target == "edge")
    {
      if(!is.null(classif_names)){classif_outcomes <- pmap(list(classif_names, group_init[fct_types], group_end[fct_types]), ~ self[[..1]](edge_features[, ..2:..3, drop = FALSE]))}
      if(!is.null(regr_names)){regr_outcomes <- pmap(list(regr_names, group_init[num_types], group_end[num_types]), ~ self[[..1]](edge_features[, ..2:..3, drop = FALSE]))}
    }

    if(target == "context")
    {
      if(!is.null(classif_names)){classif_outcomes <- pmap(list(classif_names, group_init[fct_types], group_end[fct_types]), ~ self[[..1]](context_features[, ..2:..3, drop = FALSE]))}
      if(!is.null(regr_names)){regr_outcomes <- pmap(list(regr_names, group_init[num_types], group_end[num_types]), ~ self[[..1]](context_features[, ..2:..3, drop = FALSE]))}
    }

    outcome <- vector("list", length(group_init))
    outcome[fct_types] <- classif_outcomes
    outcome[num_types] <- regr_outcomes

    return(outcome)
  })


###
training_function <- function(model, direction, node_features, edge_features, edgelist, node_train_idx, node_test_idx, context_features,
                              target, target_schema, optimization = "adam", weight_decay = 0.01, lr = 0.01, epochs = 30, patience = 10, verbose = TRUE, dev, update_order, skip_shortcut)
{
  if(optimization == "adadelta"){optimizer <- optim_adadelta(model$parameters, lr = lr, rho = 0.9, eps = 1e-06, weight_decay = weight_decay )}
  if(optimization == "adagrad"){optimizer <- optim_adagrad(model$parameters, lr = lr, lr_decay = 0, weight_decay = weight_decay , initial_accumulator_value = 0, eps = 1e-10)}
  if(optimization == "rmsprop"){optimizer <- optim_rmsprop(model$parameters, lr = lr, alpha = 0.99, eps = 1e-08, weight_decay = weight_decay , momentum = 0, centered = FALSE)}
  if(optimization == "rprop"){optimizer <- optim_rprop(model$parameters, lr = lr, etas = c(0.5, 1.2), step_sizes = c(1e-06, 50))}
  if(optimization == "sgd"){optimizer <- optim_sgd(model$parameters, lr = lr, momentum = 0, dampening = 0, weight_decay = weight_decay , nesterov = FALSE)}
  if(optimization == "asgd"){optimizer <- optim_asgd(model$parameters, lr = lr, lambda = 1e-04, alpha = 0.75, t0 = 1e+06, weight_decay = weight_decay )}
  if(optimization == "adam"){optimizer <- optim_adam(model$parameters, lr = lr, betas = c(0.9, 0.999), eps = 1e-08, weight_decay = weight_decay , amsgrad = FALSE)}

  train_history <- vector(mode="numeric", length = epochs)
  val_history <- vector(mode="numeric", length = epochs)
  dynamic_overfit <- vector(mode="numeric", length = epochs)

  node_train_idx <- sort(node_train_idx)###ORDER IS IMPORTANT FOR REINDEXING EDGELIST
  node_train <- node_features[node_train_idx,, drop=FALSE]
  edge_train_index <- graph_filter(node_train_idx, edgelist)
  edge_train_idx <- (1:nrow(edge_features))[edge_train_index]
  edge_train <- edge_features[edge_train_idx,, drop = FALSE]
  edgelist_train <- edgelist_indexer(edgelist[edge_train_index,, drop = FALSE])

  node_test_idx <- sort(node_test_idx)###ORDER IS IMPORTANT FOR REINDEXING EDGELIST
  node_val_idx <- sort(c(node_train_idx, node_test_idx))###MERGING AND ORDERING TRAIN & VALIDATION INDEXES
  node_val <- node_features[node_val_idx,, drop=FALSE]
  edge_val_index <- graph_filter(node_val_idx, edgelist)
  edge_val_idx <- (1:nrow(edge_features))[edge_val_index]
  edge_val <- edge_features[edge_val_idx,, drop = FALSE]
  edgelist_val <- edgelist_indexer(edgelist[edge_val_index,, drop = FALSE])

  delta_node_index <- !(node_val_idx %in% node_train_idx)
  delta_edge_index <- !(edge_val_idx %in% edge_train_idx)

  train_dir_list <- switch(direction, "undirected" = list(edgelist_train[,1], edgelist_train[,2]), "from_head" = list(edgelist_train[,1], NULL), "from_tail" = list(NULL, edgelist_train[,2]))
  val_dir_list <- switch(direction, "undirected" = list(edgelist_val[,1], edgelist_val[,2]), "from_head" = list(edgelist_val[,1], NULL), "from_tail" = list(NULL, edgelist_val[,2]))

  if(is.numeric(nrow(node_train)) && nrow(node_train) < 2){stop("not enough data for training")}
  if(is.numeric(nrow(edge_train)) && nrow(edge_train) < 2){stop("not enough data for training")}
  if(is.numeric(nrow(node_val)) && nrow(node_val) < 2){stop("not enough data for validation")}
  if(is.numeric(nrow(edge_val)) && nrow(edge_val) < 2){stop("not enough data for validation")}

  for(t in 1:epochs)
  {
    pred_train <- torch_cat(model(node_train, edge_train, context_features, from_node = train_dir_list[[1]], to_node = train_dir_list[[2]], update_order, skip_shortcut), dim = 2)
    actual_train <- torch_tensor(switch(target, "node" = node_train, "edge"= edge_train, "context" = context_features), dtype = torch_float(), device = dev)
    train_loss <- weighted_normalized_loss(pred_train, actual_train, target_schema, dev)
    train_history[t] <- train_loss$item()

    pred_val <- torch_cat(model(node_val, edge_val, context_features, from_node = val_dir_list[[1]], to_node = val_dir_list[[2]], update_order, skip_shortcut), dim = 2)
    actual_val <- torch_tensor(switch(target, "node" = node_val[delta_node_index,,drop = FALSE], "edge" = edge_val[delta_edge_index,,drop=FALSE], "context" = context_features), dtype = torch_float(), device = dev)
    val_filter <- switch(target, "node" = delta_node_index, "edge" = delta_edge_index, "context" = 1)
    val_loss <- weighted_normalized_loss(pred_val[val_filter,,drop = FALSE], actual_val, target_schema, dev)
    val_history[t] <- val_loss$item()

    dynamic_overfit[t] <- abs(val_history[t] - train_history[t])/abs(val_history[1] - train_history[1])

    if(verbose==TRUE){cat("epoch: ", t, "   Train loss: ", train_loss$item(), "   Val loss: ", val_loss$item(), "\n")}

    optimizer$zero_grad()
    train_loss$backward()
    optimizer$step()

    dyn_ovft_horizon <- c(0, diff(dynamic_overfit[1:t]))
    val_hist_horizon <- c(0, diff(val_history[1:t]))

    next_t <- 0
    if(t >= patience){
      loess_mod1 <- suppressWarnings(loess(h ~ t, data.frame(t=1:t, h=dyn_ovft_horizon), surface="direct", span = 1.25, degree = 1))
      next_dyn_ovft_deriv <- suppressWarnings(predict(loess_mod1, data.frame(t=t+1)))

      loess_mod2 <- suppressWarnings(loess(h ~ t, data.frame(t=1:t, h=val_hist_horizon), surface="direct", span = 1.25, degree = 1))
      next_val_loss_deriv <- suppressWarnings(predict(loess_mod2, data.frame(t=t+1)))

      rolling_window <- max(c(patience - t + 1, 1))
      avg_dyn_ovft_deriv <- mean(tail(dyn_ovft_horizon, rolling_window), na.rm = TRUE)
      avg_val_hist_deriv <- mean(tail(val_hist_horizon, rolling_window), na.rm = TRUE)
    }

    if(t >= patience && avg_dyn_ovft_deriv > 0 && next_dyn_ovft_deriv > 0 && avg_val_hist_deriv > 0 && next_val_loss_deriv > 0){if(verbose==TRUE){cat("early stop at epoch: ", t, "   Train loss: ", train_loss$item(), "   Val loss: ", val_loss$item(), "\n")}; break}
  }

  outcome <- list(model = model, train_history = train_history[1:t], val_history = val_history[1:t])

  return(outcome)
}

###
graph_filter <- function(node_idx, edge_list, mode = "&")
{
  if(mode == "&"){idx <- (edge_list[, 1] %in% node_idx) & (edge_list[, 2] %in% node_idx)}
  if(mode == "|"){idx <- (edge_list[, 1] %in% node_idx) | (edge_list[, 2] %in% node_idx)}
  return(idx)
}

###
edgelist_indexer <- function(edge_list)
{
  old_idx <- unique(as.vector(edge_list))
  new_idx <- 1:length(old_idx)
  names(new_idx) <- old_idx
  edge_list <- cbind(new_idx[as.character(edge_list[,1])], new_idx[as.character(edge_list[,2])])
  rownames(edge_list) <- NULL
  return(edge_list)
}

###
fast_naimp <- function(df, seed = 1976)
{
  set.seed(seed)

  if(!anyNA(df)){return(df)}
  if(!is.data.frame(df)){df <- as.data.frame(df)}
  feat_names <- colnames(df)
  n_rows <- nrow(df)
  if(n_rows == 1){df <- as.data.frame(t(df))}

  nan_to_na <- map(df, ~ replace(.x, is.nan(.x)|is.infinite(.x), NA))
  mask_missing <- lapply(nan_to_na, is.na)
  col_miss <- map(mask_missing, ~ sum(.x))
  imputed_values <- pmap(list(df, mask_missing, col_miss), ~ sample(..1[!..2], size = ..3, replace = TRUE))
  na_to_value <- map2(nan_to_na, imputed_values, ~ replace(.x, is.na(.x), .y))
  df <- as.data.frame(na_to_value)
  if(n_rows == 1){df <- as.data.frame(t(df))}
  colnames(df) <- feat_names

  return(df)
}

any_fct <- function(df){any(unlist(lapply(df, is.factor)))}
any_char <- function(df){any(unlist(lapply(df, is.character)))}
any_lgl <- function(df){any(unlist(lapply(df, is.logical)))}
any_num <- function(df){any(unlist(lapply(df, is.numeric)))}


how_many_fct <- function(df){sum(unlist(lapply(df, is.factor)))}
how_many_char <- function(df){sum(unlist(lapply(df, is.character)))}
how_many_lgl <- function(df){sum(unlist(lapply(df, is.logical)))}
how_many_num <- function(df){sum(unlist(lapply(df, is.numeric)))}

fct_index <- function(df){unlist(lapply(df, is.factor))}
char_index <- function(df){unlist(lapply(df, is.character))}
lgl_index <- function(df){unlist(lapply(df, is.logical))}
num_index <- function(df){unlist(lapply(df, is.numeric))}

fct_set <- function(df){all(unlist(lapply(df, is.factor)))}
char_set <- function(df){all(unlist(lapply(df, is.character)))}
lgl_set <- function(df){all(unlist(lapply(df, is.logical)))}
num_set <- function(df){all(unlist(lapply(df, is.numeric)))}

svd_reducer <- function(df, dim)
{
  dm <- data.matrix(df)
  svd_model <- svd(dm)
  redux <- svd_model$u[, 1:dim] %*% diag(svd_model$d[1:dim], dim, dim)
  colnames(redux) <- paste0("svd", 1:dim)
  return(redux)
}

###
feature_analyzer <- function(graph, labels, type, method, embedding_size, mode)
{
  if(type == "node"){n_rows <- vcount(graph)}
  if(type == "edge"){n_rows <- ecount(graph)}
  if(type == "context"){n_rows <- 1}

  if(is.character(labels))
  {
    if(type == "node"){features <- map(labels, ~ data.frame(Reduce(rbind, get.vertex.attribute(graph, .x)), stringsAsFactors = TRUE))}
    if(type == "edge"){features <- map(labels, ~ data.frame(Reduce(rbind, get.edge.attribute(graph, .x)), stringsAsFactors = TRUE))}
    if(type == "context"){features <- map(labels, ~ data.frame(matrix(unlist(graph.attributes(graph)[.x]), 1), stringsAsFactors = TRUE))}

    na_index <- map_lgl(features, ~ all(is.na(.x)))
    feat_sizes <- map_dbl(features, ~ dim(.x)[2])
    features <- pmap(list(na_index, features, feat_sizes), ~ {if(..1){as.data.frame(matrix(0, n_rows, ..3))} else {..2}})

    lgl_types <- map_lgl(features, ~ lgl_set(.x))
    fct_types <- map_lgl(features, ~ fct_set(.x))
    num_types <- map_lgl(features, ~ num_set(.x))

    features <- map_if(features, lgl_types, ~ as.data.frame(lapply(.x, as.factor)))
    fct_types <- map_lgl(features, ~ fct_set(.x))##UPDATE
    num_types <- !fct_types##UPDATE
    features <- map(features, ~ fast_naimp(.x))
    features <- map2(features, labels, ~ {if(ncol(.x)==1){colnames(.x) <- paste0(.y); return(.x)} else {colnames(.x) <- paste0(.y,"_feat", 1:ncol(.x)); return(.x)}})
    features <- map_if(features, fct_types, ~ dummy_columns(.x, remove_selected_columns = TRUE))

    group_dim <- map_dbl(features, ~ dim(.x)[2])
    features <- as.data.frame(features)
    features <- as.matrix(features)
    feat_names <- colnames(features)
    rownames(features) <- NULL

    group_end <- cumsum(group_dim)
    group_init <- group_end - group_dim + 1
  }

  if(all(is.na(labels)))
  {
    if(type == "edge"){graph <- line.graph(graph)}

    adjacency_embedding <- function(graph, embedding_size, n_rows, mode)
    {
      embed_model <- tryCatch(embed_adjacency_matrix(graph, no = embedding_size, weights = NULL), error = function(e) NA)
      if(is.na(embed_model[1])){feat_model <- matrix(0, n_rows, embedding_size); message("embedding error, using null")}
      if(!is.na(embed_model[1])){if(is.null(embed_model$Y)){feat_model <- embed_model$X} else {feat_model <- apply(abind(embed_model$X, embed_model$Y, along = 3), c(1, 2), function(x) switch(mode, "sum" = sum(x, na.rm = TRUE), "mean" = mean(x, na.rm = TRUE), "max" = max(x, na.rm = TRUE)))}}
      return(feat_model)
    }

    laplacian_embedding <- function(graph, embedding_size, n_rows, mode)
    {
      embed_model <- tryCatch(embed_laplacian_matrix(graph, no = embedding_size, weights = NULL), error = function(e) NA)
      if(is.na(embed_model[1])){feat_model <- matrix(0, n_rows, embedding_size); message("embedding error, using null")}
      if(!is.na(embed_model[1])){if(is.null(embed_model$Y)){feat_model <- embed_model$X} else {feat_model <- apply(abind(embed_model$X, embed_model$Y, along = 3), c(1, 2), function(x) switch(mode, "sum" = sum(x, na.rm = TRUE), "mean" = mean(x, na.rm = TRUE), "max" = max(x, na.rm = TRUE)))}}
      return(feat_model)
    }

    if(method == "null"){features <- matrix(0, n_rows, embedding_size)}
    if(method == "adjacency"){features <- adjacency_embedding(graph, embedding_size, n_rows, mode)}
    if(method == "laplacian"){features <- laplacian_embedding(graph, embedding_size, n_rows, mode)}
    if(type == "context"){features <- matrix(apply(features, 2, function(x) switch(mode, "sum" = sum(x), "mean" = mean(x), "max" = max(x))), 1, embedding_size)}

    if(any(!is.finite(features))){stop("not finite embedding values")}

    feat_names <- paste0("embed_feat", 1:embedding_size)
    colnames(features) <- feat_names
    group_init <- 1
    group_end <- ncol(features)
    feat_sizes <- ncol(features)
    num_types <- TRUE
    fct_types <- FALSE
   }

  outcome <- list(features = features, feat_names = feat_names, feat_sizes = feat_sizes, group_init = group_init, group_end = group_end, num_types = num_types, fct_types = fct_types)

  return(outcome)
}

###
smart_rbind <- function(mat1, mat2)
{
  feat1 <- colnames(mat1)
  feat2 <- colnames(mat2)
  if(all(feat1 %in% feat2)){out <- rbind(mat1, mat2[, feat1])}
  if(!all(feat1 %in% feat2))
  {
    missing <- setdiff(feat1, feat2)
    add_on <- matrix(0, nrow(mat2), length(missing))
    colnames(add_on) <- missing
    mat2_fixed <- cbind(mat2, add_on)
    out <- rbind(mat1, mat2_fixed[, feat1])
  }

  return(out)
}

###
weighted_normalized_loss <- function(pred, target, target_schema, dev)
{
  num_types <- target_schema$num_types
  fct_types <- target_schema$fct_types
  init <- target_schema$group_init
  end <- target_schema$group_end

  weights <- vector("list", length(init))
  if(any(fct_types)){weights[fct_types] <- map2(init[fct_types], end[fct_types],  ~ suppressWarnings(entropy(as_array(target[,.x:.y, drop = FALSE]$cpu()), method = "Laplace")))}
  if(any(num_types)){weights[num_types] <- map2(init[num_types], end[num_types],  ~ suppressWarnings(entropy(as_array(target[,.x:.y, drop = FALSE]$cpu()), method = "Laplace")))}
  entropy_weights <- unlist(weights)
  if(all(!is.finite(entropy_weights))){entropy_weights[!is.finite(entropy_weights)] <- 1}
  if(any(!is.finite(entropy_weights))){entropy_weights[!is.finite(entropy_weights)] <- mean(entropy_weights[is.finite(entropy_weights)])}

  errors <- vector("list", length(init))
  if(any(fct_types)){errors[fct_types] <- map2(init[fct_types], end[fct_types],  ~ as_array(nnf_binary_cross_entropy(pred[,.x:.y, drop = FALSE], target[,.x:.y, drop = FALSE], reduction = "mean")$cpu()))}
  if(any(num_types)){errors[num_types] <- map2(init[num_types], end[num_types],  ~ as_array(nnf_mse_loss(pred[,.x:.y, drop = FALSE], target[,.x:.y, drop = FALSE], reduction = "mean")$cpu()))}
  error_values <- map_dbl(errors, ~ 1 - 1/exp(.x))

  loss <- sum(error_values * 1/entropy_weights)/sum(1/entropy_weights)
  loss <- torch_tensor(loss, dtype = torch_float(), requires_grad = TRUE, device = dev)

  return(loss)
}

###
graph_sampling <- function(graph, samp, threshold = 0.01, seed = 42)
{
  set.seed(seed)

  g_dens <- edge_density(graph)

  if(g_dens < threshold){
    n_edge <- ecount(graph)
    if(samp > 1){sample_size <- samp}
    if(samp >= 0 & samp <= 1){sample_size <- round(n_edge * samp)}
    degree_weights <- exp(degree(line.graph(graph), normalized = T))
    eids <- sample(n_edge, sample_size, replace = FALSE, prob = degree_weights)
    sampled <- subgraph.edges(graph, eids)
  }

  if(g_dens >= threshold){
    n_nodes <- vcount(graph)
    if(samp > 1){sample_size <- samp}
    if(samp >= 0 & samp <= 1){sample_size <- round(n_nodes * samp)}
    degree_weights <- exp(degree(graph, normalized = T))
    vids <- sample(n_nodes, sample_size, replace = FALSE, prob = degree_weights)
    sampled <- subgraph(graph, vids)
  }

  return(sampled)
}

###
normalizer <- function(features, num_index)
{
  order <- colnames(features)
  num_feats <- order[num_index]
  oth_feats <- setdiff(order, num_feats)
  num_means <- apply(features[, num_feats, drop = FALSE], 2, function(x) mean(x, na.rm = TRUE))
  num_scales <- apply(features[, num_feats, drop = FALSE], 2, function(x) sd(x, na.rm = TRUE))
  if(all(num_scales == 0)){return(list(norm_features = features, num_means = NA, num_scales = NA))}
  normalized <- apply(features[, num_feats, drop = FALSE], 2, function(x) (x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
  norm_features <- cbind(features[, oth_feats, drop = FALSE], normalized)
  norm_features <- norm_features[, order, drop = FALSE]
  out <- list(norm_features = norm_features, num_means = num_means, num_scales = num_scales)
  return(out)
}

###
invert_scale <- function(scaled, center, scale, dev)
{
  if(is.list(scaled)){scaled <- scaled[[1]]}
  if("torch_tensor" %in% class(scaled)){scaled <- as.matrix(scaled$to(device = "cpu"))}
  rescaled <- sapply(1:ncol(scaled), function(i) scaled[,i] * scale[i] + center[i])
  rescaled <- torch_tensor(rescaled)$to(device = dev)
  return(rescaled)
}

