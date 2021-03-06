# Locate gdal
# This module defines
# OSG_LIBRARIES
# OSG_FOUND, if false, do not try to link to gdal 
# OSG_INCLUDE_DIR, where to find the headers
#
# $OSG_DIR is an environment variable that would
# correspond to the ./configure --prefix=$OSG_DIR
#
# Created by Robert Osfield. 

FIND_PATH(OSG_INCLUDE_DIR osg/Node
    ${OSG_DIR}/include
    $ENV{OSG_DIR}/include
    $ENV{OSG_DIR}
    $ENV{OSGDIR}/include
    $ENV{OSGDIR}
    $ENV{OSG_ROOT}/include
    NO_DEFAULT_PATH
)

MACRO(FIND_OSG_LIBRARY MYLIBRARY MYLIBRARYNAME)

    FIND_LIBRARY("${MYLIBRARY}_DEBUG"
        NAMES "${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}"
        PATHS
        ${OSG_DIR}/lib/Debug
        ${OSG_DIR}/lib64/Debug
        ${OSG_DIR}/lib
        ${OSG_DIR}/lib64
        $ENV{OSG_DIR}/lib/debug
        $ENV{OSG_DIR}/lib64/debug
        $ENV{OSG_DIR}/lib
        $ENV{OSG_DIR}/lib64
        $ENV{OSG_DIR}
        $ENV{OSGDIR}/lib
        $ENV{OSGDIR}/lib64
        $ENV{OSGDIR}
        $ENV{OSG_ROOT}/lib
        $ENV{OSG_ROOT}/lib64
        NO_DEFAULT_PATH
    )

    FIND_LIBRARY("${MYLIBRARY}_DEBUG"
        NAMES "${MYLIBRARYNAME}${CMAKE_DEBUG_POSTFIX}"
        PATHS
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OSG_ROOT]/lib
        /usr/freeware/lib64
    )
    
    FIND_LIBRARY(${MYLIBRARY}
        NAMES "${MYLIBRARYNAME}${CMAKE_RELEASE_POSTFIX}"
        PATHS
        ${OSG_DIR}/lib/Release
        ${OSG_DIR}/lib64/Release
        ${OSG_DIR}/lib
        ${OSG_DIR}/lib64
        $ENV{OSG_DIR}/lib/Release
        $ENV{OSG_DIR}/lib64/Release
        $ENV{OSG_DIR}/lib
        $ENV{OSG_DIR}/lib64
        $ENV{OSG_DIR}
        $ENV{OSGDIR}/lib
        $ENV{OSGDIR}/lib64
        $ENV{OSGDIR}
        $ENV{OSG_ROOT}/lib
        $ENV{OSG_ROOT}/lib64
        NO_DEFAULT_PATH
    )

    FIND_LIBRARY(${MYLIBRARY}
        NAMES "${MYLIBRARYNAME}${CMAKE_RELEASE_POSTFIX}"
        PATHS
        ~/Library/Frameworks
        /Library/Frameworks
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /sw/lib
        /opt/local/lib
        /opt/csw/lib
        /opt/lib
        [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OSG_ROOT]/lib
        /usr/freeware/lib64
    )
    
    IF( NOT ${MYLIBRARY}_DEBUG)
        IF(MYLIBRARY)
            SET(${MYLIBRARY}_DEBUG ${MYLIBRARY})
         ENDIF(MYLIBRARY)
    ENDIF( NOT ${MYLIBRARY}_DEBUG)
           
ENDMACRO(FIND_OSG_LIBRARY LIBRARY LIBRARYNAME)

FIND_OSG_LIBRARY(OSG_LIBRARY osg)
FIND_OSG_LIBRARY(OSGGA_LIBRARY osgGA)
FIND_OSG_LIBRARY(OSGUTIL_LIBRARY osgUtil)
FIND_OSG_LIBRARY(OSGDB_LIBRARY osgDB)
FIND_OSG_LIBRARY(OSGTEXT_LIBRARY osgText)
FIND_OSG_LIBRARY(OSGWIDGET_LIBRARY osgWidget)
FIND_OSG_LIBRARY(OSGTERRAIN_LIBRARY osgTerrain)
FIND_OSG_LIBRARY(OSGFX_LIBRARY osgFX)
FIND_OSG_LIBRARY(OSGVIEWER_LIBRARY osgViewer)
FIND_OSG_LIBRARY(OSGVOLUME_LIBRARY osgVolume)
FIND_OSG_LIBRARY(OPENTHREADS_LIBRARY OpenThreads)
FIND_OSG_LIBRARY(OSGQT_LIBRARY osgQt)
FIND_OSG_LIBRARY(OSGSHADOW_LIBRARY osgShadow)
FIND_OSG_LIBRARY(OSGSIM_LIBRARY osgSim)
FIND_OSG_LIBRARY(OSGANIMATION_LIBRARY osgAnimation)
FIND_OSG_LIBRARY(OSGMANIPULATOR_LIBRARY osgManipulator)

IF (NOT WIN32)
	IF (NOT OSG_LIBRARY_DEBUG)
		SET(OSG_LIBRARY_DEBUG ${OSG_LIBRARY})
	ENDIF()
	IF (NOT OSGDB_LIBRARY_DEBUG)
		SET(OSGDB_LIBRARY_DEBUG ${OSGDB_LIBRARY})
	ENDIF()
	IF (NOT OSGGA_LIBRARY_DEBUG)
		SET(OSGGA_LIBRARY_DEBUG ${OSGGA_LIBRARY})
	ENDIF()
	IF (NOT OSGVIEWER_LIBRARY_DEBUG)
		SET(OSGVIEWER_LIBRARY_DEBUG ${OSGVIEWER_LIBRARY})
	ENDIF()
	IF (NOT OSGUTIL_LIBRARY_DEBUG)
		SET(OSGUTIL_LIBRARY_DEBUG ${OSGUTIL_LIBRARY})
	ENDIF()
	IF (NOT OSGTEXT_LIBRARY_DEBUG)
		SET(OSGTEXT_LIBRARY_DEBUG ${OSGTEXT_LIBRARY})
	ENDIF()
	IF (NOT OPENTHREADS_LIBRARY_DEBUG)
		SET(OPENTHREADS_LIBRARY_DEBUG ${OPENTHREADS_LIBRARY})
	ENDIF()
	IF (NOT OSGWIDGET_LIBRARY_DEBUG)
		SET(OSGWIDGET_LIBRARY_DEBUG ${OSGWIDGET_LIBRARY})
	ENDIF()
	IF (NOT OSGTERRAIN_LIBRARY_DEBUG)
		SET(OSGTERRAIN_LIBRARY_DEBUG ${OSGTERRAIN_LIBRARY})
	ENDIF()
	IF (NOT OSGFX_LIBRARY_DEBUG)
		SET(OSGFX_LIBRARY_DEBUG ${OSGFX_LIBRARY})
	ENDIF()
	IF (NOT OSGVOLUME_LIBRARY_DEBUG)
		SET(OSGVOLUME_LIBRARY_DEBUG ${OSGVOLUME_LIBRARY})
	ENDIF()
	IF (NOT OSGQT_LIBRARY_DEBUG)
		SET(OSGQT_LIBRARY_DEBUG ${OSGQT_LIBRARY})
	ENDIF()
	IF (NOT OSGSHADOW_LIBRARY_DEBUG)
		SET(OSGSHADOW_LIBRARY_DEBUG ${OSGSHADOW_LIBRARY})
	ENDIF()
	IF (NOT OSGSIM_LIBRARY_DEBUG)
		SET(OSGSIM_LIBRARY_DEBUG ${OSGSIM_LIBRARY})
	ENDIF()
	IF (NOT OSGANIMATION_LIBRARY_DEBUG)
		SET(OSGANIMATION_LIBRARY_DEBUG ${OSGANIMATION_LIBRARY})
	ENDIF()
	IF (NOT OSGMANIPULATOR_LIBRARY_DEBUG)
		SET(OSGMANIPULATOR_LIBRARY_DEBUG ${OSGANIMATION_LIBRARY})
	ENDIF()
ELSE()
	IF (NOT OSG_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSG_LIBRARY_DEBUG osgd)
	ENDIF()
	IF (NOT OSGGA_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGGA_LIBRARY_DEBUG osgGAd)
	ENDIF()
	IF (NOT OSGUTIL_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGUTIL_LIBRARY_DEBUG osgUtild)
	ENDIF()
	IF (NOT OSGDB_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGDB_LIBRARY_DEBUG osgDBd)
	ENDIF()
	IF (NOT OSGTEXT_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGTEXT_LIBRARY_DEBUG osgTextd)
	ENDIF()
	IF (NOT OSGWIDGET_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGWIDGET_LIBRARY_DEBUG osgWidgetd)
	ENDIF()
	IF (NOT OSGTERRAIN_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGTERRAIN_LIBRARY_DEBUG osgTerraind)
	ENDIF()
	IF (NOT OSGFX_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGFX_LIBRARY_DEBUG osgFXd)
	ENDIF()
	IF (NOT OSGVIEWER_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGVIEWER_LIBRARY_DEBUG osgViewerd)
	ENDIF()
	IF (NOT OSGVOLUME_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGVOLUME_LIBRARY_DEBUG osgVolumed)
	ENDIF()
	IF (NOT OPENTHREADS_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OPENTHREADS_LIBRARY_DEBUG OpenThreadsd)
	ENDIF()
	IF (NOT OSGQT_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGQT_LIBRARY_DEBUG osgQtd)
	ENDIF()
	IF (NOT OSGSHADOW_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGSHADOW_LIBRARY_DEBUG osgShadowd)
	ENDIF()
	IF (NOT OSGSIM_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGSIM_LIBRARY_DEBUG osgSimd)
	ENDIF()
	IF (NOT OSGANIMATION_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGANIMATION_LIBRARY_DEBUG osgAnimationd)
	ENDIF()
	IF (NOT OSGMANIPULATOR_LIBRARY_DEBUG)
		FIND_OSG_LIBRARY(OSGMANIPULATOR_LIBRARY_DEBUG osgManipulatord)
	ENDIF()

ENDIF()

SET(OSG_FOUND "NO")
IF(OSG_LIBRARY AND OSG_INCLUDE_DIR)
    SET(OSG_FOUND "YES")
	SET(OSG_LIBRARIES 
		optimized ${OSG_LIBRARY} debug ${OSG_LIBRARY_DEBUG}
		optimized ${OSGGA_LIBRARY} debug ${OSGGA_LIBRARY_DEBUG}
		optimized ${OSGUTIL_LIBRARY} debug ${OSGUTIL_LIBRARY_DEBUG}
		optimized ${OSGDB_LIBRARY} debug ${OSGDB_LIBRARY_DEBUG}
		optimized ${OSGTEXT_LIBRARY} debug ${OSGTEXT_LIBRARY_DEBUG}
		optimized ${OSGWIDGET_LIBRARY} debug ${OSGWIDGET_LIBRARY_DEBUG}
		optimized ${OSGTERRAIN_LIBRARY} debug ${OSGTERRAIN_LIBRARY_DEBUG}
		optimized ${OSGFX_LIBRARY} debug ${OSGFX_LIBRARY_DEBUG}
		optimized ${OSGVIEWER_LIBRARY} debug ${OSGVIEWER_LIBRARY_DEBUG}
		optimized ${OSGVOLUME_LIBRARY} debug ${OSGVOLUME_LIBRARY_DEBUG}
		optimized ${OPENTHREADS_LIBRARY} debug ${OPENTHREADS_LIBRARY_DEBUG}
		optimized ${OSGQT_LIBRARY} debug ${OSGQT_LIBRARY_DEBUG}
		optimized ${OSGSHADOW_LIBRARY} debug ${OSGSHADOW_LIBRARY_DEBUG}
		optimized ${OSGSIM_LIBRARY} debug ${OSGSIM_LIBRARY_DEBUG}
		optimized ${OSGANIMATION_LIBRARY} debug ${OSGANIMATION_LIBRARY_DEBUG}
        optimized ${OSGMANIPULATOR_LIBRARY} debug ${OSGMANIPULATOR_LIBRARY_DEBUG}
	)
	SET(OSG_INCLUDE_DIRS 
		${OSG_INCLUDE_DIR}
		${OSGGA_INCLUDE_DIR}
		${OSGGUTIL_INCLUDE_DIR}
		${OSGDB_INCLUDE_DIR}
		${OSGTEXT_INCLUDE_DIR}
		${OSGWIDGET_INCLUDE_DIR}
		${OSGTERRAIN_INCLUDE_DIR}
		${OSGFX_INCLUDE_DIR}
		${OSGVIEWER_INCLUDE_DIR}
		${OSGVOLUME_INCLUDE_DIR}
		${OSGTHREADS_INCLUDE_DIR}
		${OSGQT_INCLUDE_DIR})
ENDIF(OSG_LIBRARY AND OSG_INCLUDE_DIR)


