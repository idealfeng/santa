#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

constexpr double kEps = 1e-9;

struct Point {
  double x;
  double y;
};

inline Point point_at(const double* data, npy_intp i) {
  return Point{data[2 * i + 0], data[2 * i + 1]};
}

inline double cross(const Point& o, const Point& a, const Point& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

inline bool segments_intersect(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
  const double d1 = cross(p3, p4, p1);
  const double d2 = cross(p3, p4, p2);
  const double d3 = cross(p1, p2, p3);
  const double d4 = cross(p1, p2, p4);
  const bool c1 = (d1 > kEps) != (d2 > kEps);
  const bool c2 = (d3 > kEps) != (d4 > kEps);
  return c1 && c2;
}

inline bool point_in_polygon(const Point& point, const double* poly, npy_intp n) {
  const double x = point.x;
  const double y = point.y;
  bool inside = false;
  for (npy_intp i = 0; i < n; ++i) {
    const Point p1 = point_at(poly, i);
    const Point p2 = point_at(poly, (i + 1) % n);
    const bool cond1 = (p1.y > y) != (p2.y > y);
    if (!cond1) {
      continue;
    }
    const double x_int = (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y + kEps) + p1.x;
    if (x + kEps < x_int) {
      inside = !inside;
    }
  }
  return inside;
}

inline bool polygons_intersect_impl(const double* poly1, npy_intp n1, const double* poly2, npy_intp n2) {
  // 1) AABB reject
  double min1x = std::numeric_limits<double>::infinity();
  double min1y = std::numeric_limits<double>::infinity();
  double max1x = -std::numeric_limits<double>::infinity();
  double max1y = -std::numeric_limits<double>::infinity();
  for (npy_intp i = 0; i < n1; ++i) {
    const Point p = point_at(poly1, i);
    min1x = std::min(min1x, p.x);
    min1y = std::min(min1y, p.y);
    max1x = std::max(max1x, p.x);
    max1y = std::max(max1y, p.y);
  }

  double min2x = std::numeric_limits<double>::infinity();
  double min2y = std::numeric_limits<double>::infinity();
  double max2x = -std::numeric_limits<double>::infinity();
  double max2y = -std::numeric_limits<double>::infinity();
  for (npy_intp i = 0; i < n2; ++i) {
    const Point p = point_at(poly2, i);
    min2x = std::min(min2x, p.x);
    min2y = std::min(min2y, p.y);
    max2x = std::max(max2x, p.x);
    max2y = std::max(max2y, p.y);
  }

  const bool bbox_overlap = (max1x >= min2x) && (max2x >= min1x) && (max1y >= min2y) && (max2y >= min1y);
  if (!bbox_overlap) {
    return false;
  }

  // 2) Edge intersections
  for (npy_intp i = 0; i < n1; ++i) {
    const Point a1 = point_at(poly1, i);
    const Point a2 = point_at(poly1, (i + 1) % n1);
    for (npy_intp j = 0; j < n2; ++j) {
      const Point b1 = point_at(poly2, j);
      const Point b2 = point_at(poly2, (j + 1) % n2);
      if (segments_intersect(a1, a2, b1, b2)) {
        return true;
      }
    }
  }

  // 3) Inclusion tests (match python version: test only one vertex each)
  if (n1 > 0) {
    if (point_in_polygon(point_at(poly1, 0), poly2, n2)) {
      return true;
    }
  }
  if (n2 > 0) {
    if (point_in_polygon(point_at(poly2, 0), poly1, n1)) {
      return true;
    }
  }

  return false;
}

PyArrayObject* require_poly(PyObject* obj, npy_intp* n_out) {
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  if (!arr) {
    return nullptr;
  }
  if (PyArray_NDIM(arr) != 2 || PyArray_DIM(arr, 1) != 2) {
    PyErr_SetString(PyExc_ValueError, "poly must be a 2D array with shape (N, 2) and dtype float64");
    Py_DECREF(arr);
    return nullptr;
  }
  *n_out = PyArray_DIM(arr, 0);
  return arr;
}

bool parse_point(PyObject* obj, Point* out) {
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  if (!arr) {
    return false;
  }
  bool ok = false;
  if (PyArray_NDIM(arr) == 1 && PyArray_DIM(arr, 0) == 2) {
    const double* data = reinterpret_cast<const double*>(PyArray_DATA(arr));
    *out = Point{data[0], data[1]};
    ok = true;
  } else if (PyArray_NDIM(arr) == 2 && PyArray_DIM(arr, 0) == 1 && PyArray_DIM(arr, 1) == 2) {
    const double* data = reinterpret_cast<const double*>(PyArray_DATA(arr));
    *out = Point{data[0], data[1]};
    ok = true;
  }
  Py_DECREF(arr);
  if (!ok) {
    PyErr_SetString(PyExc_ValueError, "point must be array-like with shape (2,) or (1, 2)");
  }
  return ok;
}

PyObject* py_polygons_intersect(PyObject* /*self*/, PyObject* args) {
  PyObject* poly1_obj = nullptr;
  PyObject* poly2_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &poly1_obj, &poly2_obj)) {
    return nullptr;
  }

  npy_intp n1 = 0;
  npy_intp n2 = 0;
  PyArrayObject* poly1_arr = require_poly(poly1_obj, &n1);
  if (!poly1_arr) {
    return nullptr;
  }
  PyArrayObject* poly2_arr = require_poly(poly2_obj, &n2);
  if (!poly2_arr) {
    Py_DECREF(poly1_arr);
    return nullptr;
  }

  const double* p1 = reinterpret_cast<const double*>(PyArray_DATA(poly1_arr));
  const double* p2 = reinterpret_cast<const double*>(PyArray_DATA(poly2_arr));
  const bool hit = polygons_intersect_impl(p1, n1, p2, n2);

  Py_DECREF(poly1_arr);
  Py_DECREF(poly2_arr);

  if (hit) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* py_point_in_polygon(PyObject* /*self*/, PyObject* args) {
  PyObject* point_obj = nullptr;
  PyObject* poly_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &point_obj, &poly_obj)) {
    return nullptr;
  }

  Point pt{};
  if (!parse_point(point_obj, &pt)) {
    return nullptr;
  }

  npy_intp n = 0;
  PyArrayObject* poly_arr = require_poly(poly_obj, &n);
  if (!poly_arr) {
    return nullptr;
  }
  const double* poly = reinterpret_cast<const double*>(PyArray_DATA(poly_arr));
  const bool inside = point_in_polygon(pt, poly, n);
  Py_DECREF(poly_arr);

  if (inside) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyMethodDef kMethods[] = {
    {"polygons_intersect", py_polygons_intersect, METH_VARARGS, "polygons_intersect(poly1, poly2) -> bool"},
    {"point_in_polygon", py_point_in_polygon, METH_VARARGS, "point_in_polygon(point, poly) -> bool"},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "fastcollide",
    "Fast polygon intersection helpers (C++).",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_fastcollide(void) {
  import_array();
  return PyModule_Create(&kModule);
}

