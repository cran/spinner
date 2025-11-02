set.seed(123)
g <- igraph::sample_gnp(20, 0.3, directed = FALSE, loops = FALSE)
g <- igraph::set_vertex_attr(g, "node_cat_feat",
                             value = sample(c("green","red"), 20, TRUE))
g <- igraph::set_vertex_attr(g, "node_regr_feat", value = rnorm(20, 10, 3))
g <- igraph::set_edge_attr(g, "edge_cat_feat",
                           value = sample(c("black","white","gray"),
                                          igraph::ecount(g), TRUE))
g <- igraph::set_edge_attr(g, "edge_regr_feat",
                           value = rnorm(igraph::ecount(g), 5, 1))

new_g <- igraph::add_vertices(g, 3)
new_g <- igraph::add_edges(new_g, c(1,21, 10,22, 20,23, 15,22))

test_that("node-mode: outcome format and sizes (Torch integration test)", {
  skip_on_cran()
  skip_on_os("windows")

  outcome1 <- spinner(
    g, "node",
    c("node_cat_feat","node_regr_feat"),
    c("edge_cat_feat","edge_regr_feat"),
    # if spinner exposes knobs, pass tiny values here too:
    epochs = 1L, verbose = FALSE
  )

  expect_identical(class(outcome1), "list")
  expect_length(outcome1, 8)
  expect_equal(names(outcome1),
               c("graph","model_description","model_summary","pred_fun",
                 "cv_errors","summary_errors","history","time_log"))
  expect_true(igraph::is_igraph(outcome1$graph))
  expect_true(is.function(outcome1$pred_fun))
  expect_true(is.character(outcome1$model_description))
  expect_true(is.list(outcome1$model_summary))
  expect_equal(dim(outcome1$cv_errors), c(3,4))
  expect_length(outcome1$summary_errors, 3)
  expect_true(ggplot2::is_ggplot(outcome1$history))
  expect_identical(class(outcome1$time_log)[1], "Period")

  # Prediction shape on new nodes
  pred1 <- outcome1$pred_fun(new_g)[[1]]
  expect_equal(dim(pred1), c(3,3))
})

test_that("edge-mode: outcome format and sizes (Torch integration test)", {
  skip_on_cran()
  skip_on_os("windows")

  outcome2 <- spinner(
    g, "edge",
    c("node_cat_feat","node_regr_feat"),
    c("edge_cat_feat","edge_regr_feat"),
    epochs = 1L, verbose = FALSE
  )

  expect_identical(class(outcome2), "list")
  expect_length(outcome2, 8)
  expect_equal(names(outcome2),
               c("graph","model_description","model_summary","pred_fun",
                 "cv_errors","summary_errors","history","time_log"))
  expect_true(igraph::is_igraph(outcome2$graph))
  expect_true(is.function(outcome2$pred_fun))
  expect_true(is.character(outcome2$model_description))
  expect_true(is.list(outcome2$model_summary))
  expect_equal(dim(outcome2$cv_errors), c(3,4))
  expect_length(outcome2$summary_errors, 3)
  expect_true(ggplot2::is_ggplot(outcome2$history))
  expect_identical(class(outcome2$time_log)[1], "Period")

  pred2 <- outcome2$pred_fun(new_g)[[1]]
  expect_equal(dim(pred2), c(4,5))
})

test_that("node-mode with embedding size (Torch integration test)", {
  skip_on_cran()
  skip_on_os("windows")

  outcome2bis <- spinner(g, "node", node_embedding_size = 10,
                         epochs = 1L, verbose = FALSE)

  expect_identical(class(outcome2bis), "list")
  expect_length(outcome2bis, 8)
  expect_equal(names(outcome2bis),
               c("graph","model_description","model_summary","pred_fun",
                 "cv_errors","summary_errors","history","time_log"))
  expect_true(igraph::is_igraph(outcome2bis$graph))
  expect_true(is.function(outcome2bis$pred_fun))
  expect_true(is.character(outcome2bis$model_description))
  expect_true(is.list(outcome2bis$model_summary))
  expect_equal(dim(outcome2bis$cv_errors), c(3,4))
  expect_length(outcome2bis$summary_errors, 3)
  expect_true(ggplot2::is_ggplot(outcome2bis$history))
  expect_identical(class(outcome2bis$time_log)[1], "Period")

  pred2b <- outcome2bis$pred_fun(igraph::add_vertices(g, 3))[[1]]
  expect_equal(dim(pred2b), c(3,11))
})

test_that("random search (keep=TRUE) returns expected structure", {
  skip_on_cran()
  skip_on_os("windows")

  outcome3 <- spinner_random_search(
    3L, g, "edge",
    c("node_cat_feat","node_regr_feat"),
    c("edge_cat_feat","edge_regr_feat"),
    keep = TRUE,
    epochs = 1L, verbose = FALSE
  )

  expect_true(is.data.frame(outcome3$random_search))
  expect_equal(dim(outcome3$random_search), c(3, 17))
  expect_true(is.list(outcome3$best))
  expect_length(outcome3$best, 8)
  expect_equal(dim(outcome3$best$cv_errors), c(2,4))
  expect_length(outcome3$all_models, 3)
  expect_identical(class(outcome3$time_log)[1], "Period")
})

test_that("random search (keep=FALSE) returns expected structure", {
  skip_on_cran()
  skip_on_os("windows")

  outcome4 <- spinner_random_search(
    3L, g, "edge", keep = FALSE,
    epochs = 1L, verbose = FALSE
  )

  expect_true(is.data.frame(outcome4$random_search))
  expect_equal(dim(outcome4$random_search), c(3, 19))
  expect_true(is.list(outcome4$best))
  expect_length(outcome4$best, 8)
  expect_equal(dim(outcome4$best$cv_errors), c(2,4))
  expect_false("all_models" %in% names(outcome4))
  expect_identical(class(outcome4$time_log)[1], "Period")
})
