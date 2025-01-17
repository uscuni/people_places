/*-------------------------------------------------------------------------
 *
 * pg_statistic_d.h
 *    Macro definitions for pg_statistic
 *
 * Portions Copyright (c) 1996-2024, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * NOTES
 *  ******************************
 *  *** DO NOT EDIT THIS FILE! ***
 *  ******************************
 *
 *  It has been GENERATED by src/backend/catalog/genbki.pl
 *
 *-------------------------------------------------------------------------
 */
#ifndef PG_STATISTIC_D_H
#define PG_STATISTIC_D_H

#define StatisticRelationId 2619
#define StatisticRelidAttnumInhIndexId 2696

#define Anum_pg_statistic_starelid 1
#define Anum_pg_statistic_staattnum 2
#define Anum_pg_statistic_stainherit 3
#define Anum_pg_statistic_stanullfrac 4
#define Anum_pg_statistic_stawidth 5
#define Anum_pg_statistic_stadistinct 6
#define Anum_pg_statistic_stakind1 7
#define Anum_pg_statistic_stakind2 8
#define Anum_pg_statistic_stakind3 9
#define Anum_pg_statistic_stakind4 10
#define Anum_pg_statistic_stakind5 11
#define Anum_pg_statistic_staop1 12
#define Anum_pg_statistic_staop2 13
#define Anum_pg_statistic_staop3 14
#define Anum_pg_statistic_staop4 15
#define Anum_pg_statistic_staop5 16
#define Anum_pg_statistic_stacoll1 17
#define Anum_pg_statistic_stacoll2 18
#define Anum_pg_statistic_stacoll3 19
#define Anum_pg_statistic_stacoll4 20
#define Anum_pg_statistic_stacoll5 21
#define Anum_pg_statistic_stanumbers1 22
#define Anum_pg_statistic_stanumbers2 23
#define Anum_pg_statistic_stanumbers3 24
#define Anum_pg_statistic_stanumbers4 25
#define Anum_pg_statistic_stanumbers5 26
#define Anum_pg_statistic_stavalues1 27
#define Anum_pg_statistic_stavalues2 28
#define Anum_pg_statistic_stavalues3 29
#define Anum_pg_statistic_stavalues4 30
#define Anum_pg_statistic_stavalues5 31

#define Natts_pg_statistic 31


/*
 * Several statistical slot "kinds" are defined by core PostgreSQL, as
 * documented below.  Also, custom data types can define their own "kind"
 * codes by mutual agreement between a custom typanalyze routine and the
 * selectivity estimation functions of the type's operators.
 *
 * Code reading the pg_statistic relation should not assume that a particular
 * data "kind" will appear in any particular slot.  Instead, search the
 * stakind fields to see if the desired data is available.  (The standard
 * function get_attstatsslot() may be used for this.)
 *
 * Note: The pg_stats view needs to be modified whenever a new slot kind is
 * added to core.
 */

/*
 * The present allocation of "kind" codes is:
 *
 *	1-99:		reserved for assignment by the core PostgreSQL project
 *				(values in this range will be documented in this file)
 *	100-199:	reserved for assignment by the PostGIS project
 *				(values to be documented in PostGIS documentation)
 *	200-299:	reserved for assignment by the ESRI ST_Geometry project
 *				(values to be documented in ESRI ST_Geometry documentation)
 *	300-9999:	reserved for future public assignments
 *
 * For private use you may choose a "kind" code at random in the range
 * 10000-30000.  However, for code that is to be widely disseminated it is
 * better to obtain a publicly defined "kind" code by request from the
 * PostgreSQL Global Development Group.
 */

/*
 * In a "most common values" slot, staop is the OID of the "=" operator
 * used to decide whether values are the same or not, and stacoll is the
 * collation used (same as column's collation).  stavalues contains
 * the K most common non-null values appearing in the column, and stanumbers
 * contains their frequencies (fractions of total row count).  The values
 * shall be ordered in decreasing frequency.  Note that since the arrays are
 * variable-size, K may be chosen may be chosen at ANALYZE time.  Values should
 * not appear in MCV unless they have been observed to occur more than once;
 * a unique column will have no MCV slot.
 */
#define STATISTIC_KIND_MCV	1

/*
 * A "histogram" slot describes the distribution of scalar data.  staop is
 * the OID of the "<" operator that describes the sort ordering, and stacoll
 * is the relevant collation.  (In theory more than one histogram could appear,
 * if a datatype has more than one useful sort operator or we care about more
 * than one collation.  Currently the collation will always be that of the
 * underlying column.)  stavalues contains M (>=2) non-null values that
 * divide the non-null column data values into M-1 bins of approximately equal
 * population.  The first stavalues item is the MIN and the last is the MAX.
 * stanumbers is not used and should be NULL.  IMPORTANT POINT: if an MCV
 * slot is also provided, then the histogram describes the data distribution
 * *after removing the values listed in MCV* (thus, it's a "compressed
 * histogram" in the technical parlance).  This allows a more accurate
 * representation of the distribution of a column with some very-common
 * values.  In a column with only a few distinct values, it's possible that
 * the MCV list describes the entire data population; in this case the
 * histogram reduces to empty and should be omitted.
 */
#define STATISTIC_KIND_HISTOGRAM  2

/*
 * A "correlation" slot describes the correlation between the physical order
 * of table tuples and the ordering of data values of this column, as seen
 * by the "<" operator identified by staop with the collation identified by
 * stacoll.  (As with the histogram, more than one entry could theoretically
 * appear.)  stavalues is not used and should be NULL.  stanumbers contains
 * a single entry, the correlation coefficient between the sequence of data
 * values and the sequence of their actual tuple positions.  The coefficient
 * ranges from +1 to -1.
 */
#define STATISTIC_KIND_CORRELATION	3

/*
 * A "most common elements" slot is similar to a "most common values" slot,
 * except that it stores the most common non-null *elements* of the column
 * values.  This is useful when the column datatype is an array or some other
 * type with identifiable elements (for instance, tsvector).  staop contains
 * the equality operator appropriate to the element type, and stacoll
 * contains the collation to use with it.  stavalues contains
 * the most common element values, and stanumbers their frequencies.  Unlike
 * MCV slots, frequencies are measured as the fraction of non-null rows the
 * element value appears in, not the frequency of all rows.  Also unlike
 * MCV slots, the values are sorted into the element type's default order
 * (to support binary search for a particular value).  Since this puts the
 * minimum and maximum frequencies at unpredictable spots in stanumbers,
 * there are two extra members of stanumbers, holding copies of the minimum
 * and maximum frequencies.  Optionally, there can be a third extra member,
 * which holds the frequency of null elements (expressed in the same terms:
 * the fraction of non-null rows that contain at least one null element).  If
 * this member is omitted, the column is presumed to contain no null elements.
 *
 * Note: in current usage for tsvector columns, the stavalues elements are of
 * type text, even though their representation within tsvector is not
 * exactly text.
 */
#define STATISTIC_KIND_MCELEM  4

/*
 * A "distinct elements count histogram" slot describes the distribution of
 * the number of distinct element values present in each row of an array-type
 * column.  Only non-null rows are considered, and only non-null elements.
 * staop contains the equality operator appropriate to the element type,
 * and stacoll contains the collation to use with it.
 * stavalues is not used and should be NULL.  The last member of stanumbers is
 * the average count of distinct element values over all non-null rows.  The
 * preceding M (>=2) members form a histogram that divides the population of
 * distinct-elements counts into M-1 bins of approximately equal population.
 * The first of these is the minimum observed count, and the last the maximum.
 */
#define STATISTIC_KIND_DECHIST	5

/*
 * A "length histogram" slot describes the distribution of range lengths in
 * rows of a range-type column. stanumbers contains a single entry, the
 * fraction of empty ranges. stavalues is a histogram of non-empty lengths, in
 * a format similar to STATISTIC_KIND_HISTOGRAM: it contains M (>=2) range
 * values that divide the column data values into M-1 bins of approximately
 * equal population. The lengths are stored as float8s, as measured by the
 * range type's subdiff function. Only non-null, non-empty rows are
 * considered.
 */
#define STATISTIC_KIND_RANGE_LENGTH_HISTOGRAM  6

/*
 * A "bounds histogram" slot is similar to STATISTIC_KIND_HISTOGRAM, but for
 * a range-type column.  stavalues contains M (>=2) range values that divide
 * the column data values into M-1 bins of approximately equal population.
 * Unlike a regular scalar histogram, this is actually two histograms combined
 * into a single array, with the lower bounds of each value forming a
 * histogram of lower bounds, and the upper bounds a histogram of upper
 * bounds.  Only non-NULL, non-empty ranges are included.
 */
#define STATISTIC_KIND_BOUNDS_HISTOGRAM  7


#endif							/* PG_STATISTIC_D_H */
