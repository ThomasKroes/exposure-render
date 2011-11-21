/* 
 * Here is where system computed values get stored.
 * These values should only change when the target compile platform changes.
 */

#define VTK_ER_CORE_BUILD_SHARED_LIBS
#ifndef VTK_ER_CORE_BUILD_SHARED_LIBS
#define VTK_ER_STATIC
#endif

#if defined(_MSC_VER) && !defined(VTK_ER_STATIC)
#pragma warning ( disable : 4275 )
#endif
