#ifndef SBD_CHEMISTRY_BASIC_INC_ALL_H
#define SBD_CHEMISTRY_BASIC_INC_ALL_H

#ifdef SBD_THRUST
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include<thrust/tuple.h>
#include<thrust/transform.h>
#include<thrust/iterator/zip_iterator.h>
#include <thrust/inner_product.h>
#endif

#include "sbd/chemistry/basic/integrals.h"
#ifdef SBD_THRUST
#include "sbd/chemistry/basic/integrals_thrust.h"
#endif
#include "sbd/chemistry/basic/determinants.h"
#include "sbd/chemistry/basic/helpers.h"
#include "sbd/chemistry/basic/qcham.h"
#include "sbd/chemistry/basic/correlation.h"
#include "sbd/chemistry/basic/excitation.h"
#include "sbd/chemistry/basic/makeintegrals.h"
#include "sbd/chemistry/basic/makedeterminants.h"

#endif
