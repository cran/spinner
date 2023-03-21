#' spinner_random_search
#'
#' @description spinner_random_search is a function for fine-tuning using random search on the hyper-parameter space of spinner (predefined or custom).
#'
#' @param n_samp Positive integer. Number of models to be randomly generated sampling the hyper-parameter space.
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
#' @param keep Logical. Flag to TRUE to keep all the explored models. Default: FALSE.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#'@return This function returns a list including:
#' \itemize{
#'\item random_search: summary of the sampled hyper-parameters and average error metrics.
#'\item best: best model according to overall ranking on all average error metrics (for negative metrics, absolute value is considered).
#'\item time_log: computation time.
#'\item all_models: list with all generated models (if keep flagged to TRUE).
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
#' @importFrom utils tail head
#' @importFrom stats loess predict quantile weighted.mean sd
#'
#' @references https://rpubs.com/giancarlo_vercellino/spinner
#'

###
spinner_random_search <- function(n_samp, graph, target, node_labels = NA, edge_labels = NA, context_labels = NA, direction = NULL,
                           sampling = NA, threshold = 0.01, method = NULL, node_embedding_size = NULL, edge_embedding_size = NULL, context_embedding_size = NULL,
                           update_order = NULL, n_layers = NULL, skip_shortcut = NULL, forward_layer = NULL, forward_activation = NULL, forward_drop = NULL, mode = NULL,
                           optimization = NULL, epochs = 100, lr = NULL, patience = 30, weight_decay = NULL,
                           reps = 1, folds = 2, holdout = 0.2, verbose = TRUE, seed = 42, keep = FALSE)
{
  tic.clearlog()
  tic("random search")

  set.seed(seed)

  dir_param <- sampler(direction, n_samp, range = c("from_head", "from_tail"))
  met_param <- sampler(method, n_samp, range = c("null", "laplacian", "adjacency"))
  upo_param <- sampler(update_order, n_samp, range = c("enc", "nec", "cen", "ecn", "nce", "cne"))
  lyr_param <- sampler(n_layers, n_samp, range = 1:diameter(graph), integer = T)
  ssc_param <- sampler(skip_shortcut, n_samp, range = c(T, F), integer = T)
  flr_param <- sampler(forward_layer, n_samp, range = 8:1024, integer = T)
  act_param <- sampler(forward_activation, n_samp, range = c("linear", "relu", "mish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "sigmoid", "tanh"), integer = F)
  drp_param <- sampler(forward_drop, n_samp, range = c(0, 1), integer = F)
  md_param <- sampler(mode, n_samp, range = c("sum", "mean", "max"))
  opt_param <- sampler(optimization, n_samp, range = c("adam", "adagrad", "adadelta", "sgd", "asgd", "rprop", "rmsprop"))
  nd_embd_param <- sampler(node_embedding_size, n_samp, range = 2:10, integer = T)
  edg_embd_param <- sampler(edge_embedding_size, n_samp, range = 2:10, integer = T)
  ctx_embd_param <- sampler(context_embedding_size, n_samp, range = 2:10, integer = T)
  lr_param <- sampler(lr, n_samp, range = c(0, 0.1), integer = F)
  wdec_param <- sampler(weight_decay, n_samp, range = c(0, 0.1), integer = F)

  hyper_params <- list(dir_param, met_param, upo_param, lyr_param, ssc_param, flr_param, act_param, drp_param, md_param, opt_param, lr_param, wdec_param)
  if(all(is.na(node_labels))){hyper_params[[13]] <- nd_embd_param} else {hyper_params[[13]] <- rep(5, n_samp)}###DUMMY VALUE NEEDED EVEN IF NOT USED AT ALL
  if(all(is.na(edge_labels))){hyper_params[[14]] <- edg_embd_param} else {hyper_params[[14]] <- rep(5, n_samp)}###DUMMY VALUE NEEDED EVEN IF NOT USED AT ALL
  if(all(is.na(context_labels))){hyper_params[[15]] <- ctx_embd_param} else {hyper_params[[15]] <- rep(5, n_samp)}###DUMMY VALUE NEEDED EVEN IF NOT USED AT ALL

  models <- purrr::pmap(hyper_params, ~ spinner(graph, target, node_labels, edge_labels, context_labels, direction = ..1,
                                                sampling = NA, threshold = 0.01, method = ..2, node_embedding_size = ..13, edge_embedding_size = ..14, context_embedding_size = ..15,
                                                update_order = ..3, n_layers = ..4, skip_shortcut = ..5, forward_layer = ..6, forward_activation = ..7, forward_drop = ..8, mode = ..9,
                                                optimization = ..10, epochs, lr =..11, patience, weight_decay = ..12, reps, folds, holdout, verbose, seed))

  random_search <- data.frame(model = 1:n_samp)
  random_search$direction <- dir_param
  random_search$method <- met_param
  random_search$update_order <- upo_param
  random_search$n_layers <- lyr_param
  random_search$skip_shortcut <- ssc_param
  random_search$forward_layer <- flr_param
  random_search$forward_activation <- act_param
  random_search$forward_drop <- drp_param
  random_search$mode <- md_param
  random_search$optimization <- opt_param
  random_search$lr <- lr_param
  random_search$weight_decay <- wdec_param
  random_search$node_embedding_size <- hyper_params[[13]]
  random_search$edge_embedding_size <- hyper_params[[14]]
  random_search$context_embedding_size <- hyper_params[[15]]

  errors <- t(as.data.frame(map(models, ~.x$summary_errors)))
  colnames(errors) <- c("train", "validation", "test")
  rownames(errors) <- NULL

  random_search <- as.data.frame(cbind(random_search, errors))
  random_search <- ranker(random_search, 17:19, weights = 1:3)
  best <- models[[head(random_search$model, 1)]]

  if(all(!is.na(node_labels))){random_search <- random_search[,setdiff(colnames(random_search), "node_embedding_size")]}
  if(all(!is.na(edge_labels))){random_search <- random_search[,setdiff(colnames(random_search), "edge_embedding_size")]}
  if(all(!is.na(context_labels))){random_search <- random_search[,setdiff(colnames(random_search), "context_embedding_size")]}

  toc(log = TRUE)
  time_log <- tail(seconds_to_period(round(parse_number(unlist(tic.log())), 0)), 1)

  outcome <- list(random_search = random_search, best = best, time_log = time_log)
  if(keep){outcome$all_models <- models}

  return(outcome)
}

###
sampler <- function(vect, n_samp, range = NULL, integer = FALSE, multi = NULL, variable = NULL, similar = NULL)
{
  if(is.null(vect))
  {
    if(!is.character(range)){if(integer){set <- min(range):max(range)} else {set <- seq(min(range), max(range), length.out = 1000)}} else {set <- range}
    if(is.null(multi) & is.null(variable) & is.null(similar)){samp <- sample(set, n_samp, replace = TRUE)}
    if(is.numeric(multi) & is.null(variable) & is.null(similar)){samp <- replicate(n_samp, sample(set, multi, replace = TRUE), simplify = FALSE)}
    if(is.numeric(variable) & is.null(multi) & is.null(similar)){samp <- replicate(n_samp, sample(set, sample(variable, 1), replace = TRUE), simplify = FALSE)}
    if(!is.null(similar) & is.null(multi) & is.null(variable)){samp <- map(similar, ~ sample(set, length(.x), replace = TRUE), simplify = FALSE)}
  }

  if(!is.null(vect))
  {
    if(is.null(multi) & is.null(variable) & is.null(similar)){
      if(length(vect)==1){samp <- rep(vect, n_samp)}
      if(length(vect) > 1){samp <- sample(vect, n_samp, replace = TRUE)}
    }

    if(is.numeric(multi) & is.null(variable) & is.null(similar)){
      if(length(vect)==1){samp <- replicate(n_samp, rep(vect, multi), simplify = FALSE)}
      if(length(vect) > 1){samp <- replicate(n_samp, sample(vect, multi, replace = TRUE), simplify = FALSE)}
    }

    if(is.numeric(variable)& is.null(multi) & is.null(similar)){samp <- replicate(n_samp, sample(vect, sample(variable, 1), replace = TRUE), simplify = FALSE)}
    if(!is.null(similar) & is.null(multi) & is.null(variable)){samp <- map(similar, ~ sample(vect, length(.x), replace = TRUE), simplify = FALSE)}
  }

  return(samp)
}

###
ranker <- function(df, focus, inverse = NULL, absolute = NULL, reverse = FALSE, weights = NULL)
{
  rank_set <- df[, focus, drop = FALSE]
  if(!is.null(inverse)){rank_set[, inverse] <- - rank_set[, inverse]}###INVERSION BY COL NAMES
  if(!is.null(absolute)){rank_set[, absolute] <- abs(rank_set[, absolute])}###ABS BY COL NAMES
  if(is.null(weights)){index <- apply(scale(rank_set), 1, mean, na.rm = TRUE)}
  if(!is.null(weights)){index <- apply(scale(rank_set), 1, function(x) weighted.mean(x, weights, na.rm = TRUE))}
  if(reverse == FALSE){df <- df[order(index),]}
  if(reverse == TRUE){df <- df[order(-index),]}
  return(df)
}

