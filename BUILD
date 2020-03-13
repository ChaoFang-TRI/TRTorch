load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include_core",
    package_dir = "include/trtorch",
    deps = [
        "//core:include",
        "//core/conversion:include",
        "//core/conversion/conversionctx:include",
        "//core/conversion/converters:include",
        "//core/conversion/evaluators:include",
        "//core/execution:include",
        "//core/lowering:include",
        "//core/lowering/irfusers:include",
        "//core/util:include",
        "//core/util/logging:include"
    ],
)

pkg_tar(
    name = "include",
    package_dir = "include/trtorch/",
    srcs = [
        "//cpp/api:api_headers",
    ],
)

pkg_tar(
    name = "lib",
    package_dir = "lib/",
    srcs = [
        "//cpp/api/lib:libtrtorch.so",
    ],
    mode = "0755",
)




pkg_tar(
    name = "libtrtorch",
    extension = "tar.gz",
    package_dir = "trtorch",
    srcs = [
        "//:LICENSE"
    ],
    deps = [
        ":lib",
        ":include",
        ":include_core",
    ],
)
