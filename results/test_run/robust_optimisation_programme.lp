\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_HSBC:                       w_0_HSBC - 0.99 b_0_HSBC + 1.01 s_0_HSBC
                                  = 0.1
 w_0_VOD:                        w_0_VOD - 0.99 b_0_VOD + 1.01 s_0_VOD  = 0.1
 w_0_BP:                         w_0_BP - 0.99 b_0_BP + 1.01 s_0_BP  = 0.1
 w_0_GSK:                        w_0_GSK - 0.99 b_0_GSK + 1.01 s_0_GSK  = 0.1
 w_0_AZN:                        w_0_AZN - 0.99 b_0_AZN + 1.01 s_0_AZN  = 0.1
 w_0_RIO:                        w_0_RIO - 0.99 b_0_RIO + 1.01 s_0_RIO  = 0.1
 w_0_BG:                         w_0_BG - 0.99 b_0_BG + 1.01 s_0_BG  = 0.1
 w_0_TSCO:                       w_0_TSCO - 0.99 b_0_TSCO + 1.01 s_0_TSCO
                                  = 0.1
 w_0_AAL:                        w_0_AAL - 0.99 b_0_AAL + 1.01 s_0_AAL  = 0.1
 w_0_PRU:                        w_0_PRU - 0.99 b_0_PRU + 1.01 s_0_PRU  = 0.1
 w_0_balance_constraint:         b_0_HSBC + b_0_VOD + b_0_BP + b_0_GSK
                                 + b_0_AZN + b_0_RIO + b_0_BG + b_0_TSCO
                                 + b_0_AAL + b_0_PRU - s_0_HSBC - s_0_VOD
                                 - s_0_BP - s_0_GSK - s_0_AZN - s_0_RIO
                                 - s_0_BG - s_0_TSCO - s_0_AAL - s_0_PRU  = 0
 w_k10_HSBC_s0_return_constraint: 1.14085 w_0_HSBC - w_k10_HSBC_s0
                                 + 0.99 b_k10_HSBC_s0 - 1.01 s_k10_HSBC_s0  = 0
 w_k10_HSBC_s1_return_constraint: 0.882863 w_0_HSBC - w_k10_HSBC_s1
                                 + 0.99 b_k10_HSBC_s1 - 1.01 s_k10_HSBC_s1  = 0
 w_k10_HSBC_s2_return_constraint: 1.09003 w_0_HSBC - w_k10_HSBC_s2
                                 + 0.99 b_k10_HSBC_s2 - 1.01 s_k10_HSBC_s2  = 0
 w_k10_HSBC_s3_return_constraint: 0.949047 w_0_HSBC - w_k10_HSBC_s3
                                 + 0.99 b_k10_HSBC_s3 - 1.01 s_k10_HSBC_s3  = 0
 w_k10_HSBC_s4_return_constraint: 0.988452 w_0_HSBC - w_k10_HSBC_s4
                                 + 0.99 b_k10_HSBC_s4 - 1.01 s_k10_HSBC_s4  = 0
 w_k10_HSBC_s5_return_constraint: 0.950952 w_0_HSBC - w_k10_HSBC_s5
                                 + 0.99 b_k10_HSBC_s5 - 1.01 s_k10_HSBC_s5  = 0
 w_k10_HSBC_s6_return_constraint: 0.870452 w_0_HSBC - w_k10_HSBC_s6
                                 + 0.99 b_k10_HSBC_s6 - 1.01 s_k10_HSBC_s6  = 0
 w_k10_HSBC_s7_return_constraint: 1.01371 w_0_HSBC - w_k10_HSBC_s7
                                 + 0.99 b_k10_HSBC_s7 - 1.01 s_k10_HSBC_s7  = 0
 w_k10_HSBC_s8_return_constraint: 1.03885 w_0_HSBC - w_k10_HSBC_s8
                                 + 0.99 b_k10_HSBC_s8 - 1.01 s_k10_HSBC_s8  = 0
 w_k10_HSBC_s9_return_constraint: 0.950247 w_0_HSBC - w_k10_HSBC_s9
                                 + 0.99 b_k10_HSBC_s9 - 1.01 s_k10_HSBC_s9  = 0
 w_k10_HSBC_s00_return_constraint: 0.912231 w_k10_HSBC_s0 - w_k10_HSBC_s00
                                 + 0.99 b_k10_HSBC_s00 - 1.01 s_k10_HSBC_s00
                                  = 0
 w_k10_HSBC_s01_return_constraint: 1.08374 w_k10_HSBC_s0 - w_k10_HSBC_s01
                                 + 0.99 b_k10_HSBC_s01 - 1.01 s_k10_HSBC_s01
                                  = 0
 w_k10_HSBC_s02_return_constraint: 1.07671 w_k10_HSBC_s0 - w_k10_HSBC_s02
                                 + 0.99 b_k10_HSBC_s02 - 1.01 s_k10_HSBC_s02
                                  = 0
 w_k10_HSBC_s03_return_constraint: 1.32935 w_k10_HSBC_s0 - w_k10_HSBC_s03
                                 + 0.99 b_k10_HSBC_s03 - 1.01 s_k10_HSBC_s03
                                  = 0
 w_k10_HSBC_s04_return_constraint: 1.02231 w_k10_HSBC_s0 - w_k10_HSBC_s04
                                 + 0.99 b_k10_HSBC_s04 - 1.01 s_k10_HSBC_s04
                                  = 0
 w_k10_HSBC_s05_return_constraint: 0.921817 w_k10_HSBC_s0 - w_k10_HSBC_s05
                                 + 0.99 b_k10_HSBC_s05 - 1.01 s_k10_HSBC_s05
                                  = 0
 w_k10_HSBC_s06_return_constraint: 0.962573 w_k10_HSBC_s0 - w_k10_HSBC_s06
                                 + 0.99 b_k10_HSBC_s06 - 1.01 s_k10_HSBC_s06
                                  = 0
 w_k10_HSBC_s07_return_constraint: 1.10321 w_k10_HSBC_s0 - w_k10_HSBC_s07
                                 + 0.99 b_k10_HSBC_s07 - 1.01 s_k10_HSBC_s07
                                  = 0
 w_k10_HSBC_s08_return_constraint: 1.16668 w_k10_HSBC_s0 - w_k10_HSBC_s08
                                 + 0.99 b_k10_HSBC_s08 - 1.01 s_k10_HSBC_s08
                                  = 0
 w_k10_HSBC_s09_return_constraint: 1.10694 w_k10_HSBC_s0 - w_k10_HSBC_s09
                                 + 0.99 b_k10_HSBC_s09 - 1.01 s_k10_HSBC_s09
                                  = 0
 w_k10_HSBC_s10_return_constraint: 1.16927 w_k10_HSBC_s1 - w_k10_HSBC_s10
                                 + 0.99 b_k10_HSBC_s10 - 1.01 s_k10_HSBC_s10
                                  = 0
 w_k10_HSBC_s11_return_constraint: 0.956634 w_k10_HSBC_s1 - w_k10_HSBC_s11
                                 + 0.99 b_k10_HSBC_s11 - 1.01 s_k10_HSBC_s11
                                  = 0
 w_k10_HSBC_s12_return_constraint: 1.05985 w_k10_HSBC_s1 - w_k10_HSBC_s12
                                 + 0.99 b_k10_HSBC_s12 - 1.01 s_k10_HSBC_s12
                                  = 0
 w_k10_HSBC_s13_return_constraint: 0.790144 w_k10_HSBC_s1 - w_k10_HSBC_s13
                                 + 0.99 b_k10_HSBC_s13 - 1.01 s_k10_HSBC_s13
                                  = 0
 w_k10_HSBC_s14_return_constraint: 1.07204 w_k10_HSBC_s1 - w_k10_HSBC_s14
                                 + 0.99 b_k10_HSBC_s14 - 1.01 s_k10_HSBC_s14
                                  = 0
 w_k10_HSBC_s15_return_constraint: 1.23761 w_k10_HSBC_s1 - w_k10_HSBC_s15
                                 + 0.99 b_k10_HSBC_s15 - 1.01 s_k10_HSBC_s15
                                  = 0
 w_k10_HSBC_s16_return_constraint: 0.918033 w_k10_HSBC_s1 - w_k10_HSBC_s16
                                 + 0.99 b_k10_HSBC_s16 - 1.01 s_k10_HSBC_s16
                                  = 0
 w_k10_HSBC_s17_return_constraint: 0.946137 w_k10_HSBC_s1 - w_k10_HSBC_s17
                                 + 0.99 b_k10_HSBC_s17 - 1.01 s_k10_HSBC_s17
                                  = 0
 w_k10_HSBC_s18_return_constraint: 0.884912 w_k10_HSBC_s1 - w_k10_HSBC_s18
                                 + 0.99 b_k10_HSBC_s18 - 1.01 s_k10_HSBC_s18
                                  = 0
 w_k10_HSBC_s19_return_constraint: 0.795758 w_k10_HSBC_s1 - w_k10_HSBC_s19
                                 + 0.99 b_k10_HSBC_s19 - 1.01 s_k10_HSBC_s19
                                  = 0
 w_k10_HSBC_s20_return_constraint: 0.852736 w_k10_HSBC_s2 - w_k10_HSBC_s20
                                 + 0.99 b_k10_HSBC_s20 - 1.01 s_k10_HSBC_s20
                                  = 0
 w_k10_HSBC_s21_return_constraint: 0.871712 w_k10_HSBC_s2 - w_k10_HSBC_s21
                                 + 0.99 b_k10_HSBC_s21 - 1.01 s_k10_HSBC_s21
                                  = 0
 w_k10_HSBC_s22_return_constraint: 1.1192 w_k10_HSBC_s2 - w_k10_HSBC_s22
                                 + 0.99 b_k10_HSBC_s22 - 1.01 s_k10_HSBC_s22
                                  = 0
 w_k10_HSBC_s23_return_constraint: 1.11978 w_k10_HSBC_s2 - w_k10_HSBC_s23
                                 + 0.99 b_k10_HSBC_s23 - 1.01 s_k10_HSBC_s23
                                  = 0
 w_k10_HSBC_s24_return_constraint: 1.23525 w_k10_HSBC_s2 - w_k10_HSBC_s24
                                 + 0.99 b_k10_HSBC_s24 - 1.01 s_k10_HSBC_s24
                                  = 0
 w_k10_HSBC_s25_return_constraint: 0.923665 w_k10_HSBC_s2 - w_k10_HSBC_s25
                                 + 0.99 b_k10_HSBC_s25 - 1.01 s_k10_HSBC_s25
                                  = 0
 w_k10_HSBC_s26_return_constraint: 0.976287 w_k10_HSBC_s2 - w_k10_HSBC_s26
                                 + 0.99 b_k10_HSBC_s26 - 1.01 s_k10_HSBC_s26
                                  = 0
 w_k10_HSBC_s27_return_constraint: 1.09444 w_k10_HSBC_s2 - w_k10_HSBC_s27
                                 + 0.99 b_k10_HSBC_s27 - 1.01 s_k10_HSBC_s27
                                  = 0
 w_k10_HSBC_s28_return_constraint: 1.08748 w_k10_HSBC_s2 - w_k10_HSBC_s28
                                 + 0.99 b_k10_HSBC_s28 - 1.01 s_k10_HSBC_s28
                                  = 0
 w_k10_HSBC_s29_return_constraint: 0.897664 w_k10_HSBC_s2 - w_k10_HSBC_s29
                                 + 0.99 b_k10_HSBC_s29 - 1.01 s_k10_HSBC_s29
                                  = 0
 w_k10_HSBC_s30_return_constraint: 1.08952 w_k10_HSBC_s3 - w_k10_HSBC_s30
                                 + 0.99 b_k10_HSBC_s30 - 1.01 s_k10_HSBC_s30
                                  = 0
 w_k10_HSBC_s31_return_constraint: 0.951363 w_k10_HSBC_s3 - w_k10_HSBC_s31
                                 + 0.99 b_k10_HSBC_s31 - 1.01 s_k10_HSBC_s31
                                  = 0
 w_k10_HSBC_s32_return_constraint: 0.984716 w_k10_HSBC_s3 - w_k10_HSBC_s32
                                 + 0.99 b_k10_HSBC_s32 - 1.01 s_k10_HSBC_s32
                                  = 0
 w_k10_HSBC_s33_return_constraint: 0.879312 w_k10_HSBC_s3 - w_k10_HSBC_s33
                                 + 0.99 b_k10_HSBC_s33 - 1.01 s_k10_HSBC_s33
                                  = 0
 w_k10_HSBC_s34_return_constraint: 1.41297 w_k10_HSBC_s3 - w_k10_HSBC_s34
                                 + 0.99 b_k10_HSBC_s34 - 1.01 s_k10_HSBC_s34
                                  = 0
 w_k10_HSBC_s35_return_constraint: 1.12858 w_k10_HSBC_s3 - w_k10_HSBC_s35
                                 + 0.99 b_k10_HSBC_s35 - 1.01 s_k10_HSBC_s35
                                  = 0
 w_k10_HSBC_s36_return_constraint: 0.922534 w_k10_HSBC_s3 - w_k10_HSBC_s36
                                 + 0.99 b_k10_HSBC_s36 - 1.01 s_k10_HSBC_s36
                                  = 0
 w_k10_HSBC_s37_return_constraint: 1.0835 w_k10_HSBC_s3 - w_k10_HSBC_s37
                                 + 0.99 b_k10_HSBC_s37 - 1.01 s_k10_HSBC_s37
                                  = 0
 w_k10_HSBC_s38_return_constraint: 1.02364 w_k10_HSBC_s3 - w_k10_HSBC_s38
                                 + 0.99 b_k10_HSBC_s38 - 1.01 s_k10_HSBC_s38
                                  = 0
 w_k10_HSBC_s39_return_constraint: 0.880454 w_k10_HSBC_s3 - w_k10_HSBC_s39
                                 + 0.99 b_k10_HSBC_s39 - 1.01 s_k10_HSBC_s39
                                  = 0
 w_k10_HSBC_s40_return_constraint: 0.956059 w_k10_HSBC_s4 - w_k10_HSBC_s40
                                 + 0.99 b_k10_HSBC_s40 - 1.01 s_k10_HSBC_s40
                                  = 0
 w_k10_HSBC_s41_return_constraint: 0.92729 w_k10_HSBC_s4 - w_k10_HSBC_s41
                                 + 0.99 b_k10_HSBC_s41 - 1.01 s_k10_HSBC_s41
                                  = 0
 w_k10_HSBC_s42_return_constraint: 1.34904 w_k10_HSBC_s4 - w_k10_HSBC_s42
                                 + 0.99 b_k10_HSBC_s42 - 1.01 s_k10_HSBC_s42
                                  = 0
 w_k10_HSBC_s43_return_constraint: 0.935657 w_k10_HSBC_s4 - w_k10_HSBC_s43
                                 + 0.99 b_k10_HSBC_s43 - 1.01 s_k10_HSBC_s43
                                  = 0
 w_k10_HSBC_s44_return_constraint: 1.02674 w_k10_HSBC_s4 - w_k10_HSBC_s44
                                 + 0.99 b_k10_HSBC_s44 - 1.01 s_k10_HSBC_s44
                                  = 0
 w_k10_HSBC_s45_return_constraint: 0.980149 w_k10_HSBC_s4 - w_k10_HSBC_s45
                                 + 0.99 b_k10_HSBC_s45 - 1.01 s_k10_HSBC_s45
                                  = 0
 w_k10_HSBC_s46_return_constraint: 1.09047 w_k10_HSBC_s4 - w_k10_HSBC_s46
                                 + 0.99 b_k10_HSBC_s46 - 1.01 s_k10_HSBC_s46
                                  = 0
 w_k10_HSBC_s47_return_constraint: 1.07547 w_k10_HSBC_s4 - w_k10_HSBC_s47
                                 + 0.99 b_k10_HSBC_s47 - 1.01 s_k10_HSBC_s47
                                  = 0
 w_k10_HSBC_s48_return_constraint: 1.08196 w_k10_HSBC_s4 - w_k10_HSBC_s48
                                 + 0.99 b_k10_HSBC_s48 - 1.01 s_k10_HSBC_s48
                                  = 0
 w_k10_HSBC_s49_return_constraint: 0.800719 w_k10_HSBC_s4 - w_k10_HSBC_s49
                                 + 0.99 b_k10_HSBC_s49 - 1.01 s_k10_HSBC_s49
                                  = 0
 w_k10_HSBC_s50_return_constraint: 1.18797 w_k10_HSBC_s5 - w_k10_HSBC_s50
                                 + 0.99 b_k10_HSBC_s50 - 1.01 s_k10_HSBC_s50
                                  = 0
 w_k10_HSBC_s51_return_constraint: 0.952294 w_k10_HSBC_s5 - w_k10_HSBC_s51
                                 + 0.99 b_k10_HSBC_s51 - 1.01 s_k10_HSBC_s51
                                  = 0
 w_k10_HSBC_s52_return_constraint: 0.867774 w_k10_HSBC_s5 - w_k10_HSBC_s52
                                 + 0.99 b_k10_HSBC_s52 - 1.01 s_k10_HSBC_s52
                                  = 0
 w_k10_HSBC_s53_return_constraint: 1.18854 w_k10_HSBC_s5 - w_k10_HSBC_s53
                                 + 0.99 b_k10_HSBC_s53 - 1.01 s_k10_HSBC_s53
                                  = 0
 w_k10_HSBC_s54_return_constraint: 1.09022 w_k10_HSBC_s5 - w_k10_HSBC_s54
                                 + 0.99 b_k10_HSBC_s54 - 1.01 s_k10_HSBC_s54
                                  = 0
 w_k10_HSBC_s55_return_constraint: 0.805176 w_k10_HSBC_s5 - w_k10_HSBC_s55
                                 + 0.99 b_k10_HSBC_s55 - 1.01 s_k10_HSBC_s55
                                  = 0
 w_k10_HSBC_s56_return_constraint: 0.939518 w_k10_HSBC_s5 - w_k10_HSBC_s56
                                 + 0.99 b_k10_HSBC_s56 - 1.01 s_k10_HSBC_s56
                                  = 0
 w_k10_HSBC_s57_return_constraint: 1.07116 w_k10_HSBC_s5 - w_k10_HSBC_s57
                                 + 0.99 b_k10_HSBC_s57 - 1.01 s_k10_HSBC_s57
                                  = 0
 w_k10_HSBC_s58_return_constraint: 0.925015 w_k10_HSBC_s5 - w_k10_HSBC_s58
                                 + 0.99 b_k10_HSBC_s58 - 1.01 s_k10_HSBC_s58
                                  = 0
 w_k10_HSBC_s59_return_constraint: 1.07452 w_k10_HSBC_s5 - w_k10_HSBC_s59
                                 + 0.99 b_k10_HSBC_s59 - 1.01 s_k10_HSBC_s59
                                  = 0
 w_k10_HSBC_s60_return_constraint: 1.00939 w_k10_HSBC_s6 - w_k10_HSBC_s60
                                 + 0.99 b_k10_HSBC_s60 - 1.01 s_k10_HSBC_s60
                                  = 0
 w_k10_HSBC_s61_return_constraint: 1.08224 w_k10_HSBC_s6 - w_k10_HSBC_s61
                                 + 0.99 b_k10_HSBC_s61 - 1.01 s_k10_HSBC_s61
                                  = 0
 w_k10_HSBC_s62_return_constraint: 1.00491 w_k10_HSBC_s6 - w_k10_HSBC_s62
                                 + 0.99 b_k10_HSBC_s62 - 1.01 s_k10_HSBC_s62
                                  = 0
 w_k10_HSBC_s63_return_constraint: 0.919462 w_k10_HSBC_s6 - w_k10_HSBC_s63
                                 + 0.99 b_k10_HSBC_s63 - 1.01 s_k10_HSBC_s63
                                  = 0
 w_k10_HSBC_s64_return_constraint: 0.806696 w_k10_HSBC_s6 - w_k10_HSBC_s64
                                 + 0.99 b_k10_HSBC_s64 - 1.01 s_k10_HSBC_s64
                                  = 0
 w_k10_HSBC_s65_return_constraint: 1.00748 w_k10_HSBC_s6 - w_k10_HSBC_s65
                                 + 0.99 b_k10_HSBC_s65 - 1.01 s_k10_HSBC_s65
                                  = 0
 w_k10_HSBC_s66_return_constraint: 1.01212 w_k10_HSBC_s6 - w_k10_HSBC_s66
                                 + 0.99 b_k10_HSBC_s66 - 1.01 s_k10_HSBC_s66
                                  = 0
 w_k10_HSBC_s67_return_constraint: 1.19153 w_k10_HSBC_s6 - w_k10_HSBC_s67
                                 + 0.99 b_k10_HSBC_s67 - 1.01 s_k10_HSBC_s67
                                  = 0
 w_k10_HSBC_s68_return_constraint: 1.13489 w_k10_HSBC_s6 - w_k10_HSBC_s68
                                 + 0.99 b_k10_HSBC_s68 - 1.01 s_k10_HSBC_s68
                                  = 0
 w_k10_HSBC_s69_return_constraint: 0.970107 w_k10_HSBC_s6 - w_k10_HSBC_s69
                                 + 0.99 b_k10_HSBC_s69 - 1.01 s_k10_HSBC_s69
                                  = 0
 w_k10_HSBC_s70_return_constraint: 0.91346 w_k10_HSBC_s7 - w_k10_HSBC_s70
                                 + 0.99 b_k10_HSBC_s70 - 1.01 s_k10_HSBC_s70
                                  = 0
 w_k10_HSBC_s71_return_constraint: 1.08476 w_k10_HSBC_s7 - w_k10_HSBC_s71
                                 + 0.99 b_k10_HSBC_s71 - 1.01 s_k10_HSBC_s71
                                  = 0
 w_k10_HSBC_s72_return_constraint: 1.00841 w_k10_HSBC_s7 - w_k10_HSBC_s72
                                 + 0.99 b_k10_HSBC_s72 - 1.01 s_k10_HSBC_s72
                                  = 0
 w_k10_HSBC_s73_return_constraint: 0.971858 w_k10_HSBC_s7 - w_k10_HSBC_s73
                                 + 0.99 b_k10_HSBC_s73 - 1.01 s_k10_HSBC_s73
                                  = 0
 w_k10_HSBC_s74_return_constraint: 0.744944 w_k10_HSBC_s7 - w_k10_HSBC_s74
                                 + 0.99 b_k10_HSBC_s74 - 1.01 s_k10_HSBC_s74
                                  = 0
 w_k10_HSBC_s75_return_constraint: 0.915026 w_k10_HSBC_s7 - w_k10_HSBC_s75
                                 + 0.99 b_k10_HSBC_s75 - 1.01 s_k10_HSBC_s75
                                  = 0
 w_k10_HSBC_s76_return_constraint: 0.89932 w_k10_HSBC_s7 - w_k10_HSBC_s76
                                 + 0.99 b_k10_HSBC_s76 - 1.01 s_k10_HSBC_s76
                                  = 0
 w_k10_HSBC_s77_return_constraint: 1.04519 w_k10_HSBC_s7 - w_k10_HSBC_s77
                                 + 0.99 b_k10_HSBC_s77 - 1.01 s_k10_HSBC_s77
                                  = 0
 w_k10_HSBC_s78_return_constraint: 0.969133 w_k10_HSBC_s7 - w_k10_HSBC_s78
                                 + 0.99 b_k10_HSBC_s78 - 1.01 s_k10_HSBC_s78
                                  = 0
 w_k10_HSBC_s79_return_constraint: 1.11495 w_k10_HSBC_s7 - w_k10_HSBC_s79
                                 + 0.99 b_k10_HSBC_s79 - 1.01 s_k10_HSBC_s79
                                  = 0
 w_k10_HSBC_s80_return_constraint: 1.13312 w_k10_HSBC_s8 - w_k10_HSBC_s80
                                 + 0.99 b_k10_HSBC_s80 - 1.01 s_k10_HSBC_s80
                                  = 0
 w_k10_HSBC_s81_return_constraint: 0.958686 w_k10_HSBC_s8 - w_k10_HSBC_s81
                                 + 0.99 b_k10_HSBC_s81 - 1.01 s_k10_HSBC_s81
                                  = 0
 w_k10_HSBC_s82_return_constraint: 1.11351 w_k10_HSBC_s8 - w_k10_HSBC_s82
                                 + 0.99 b_k10_HSBC_s82 - 1.01 s_k10_HSBC_s82
                                  = 0
 w_k10_HSBC_s83_return_constraint: 1.02588 w_k10_HSBC_s8 - w_k10_HSBC_s83
                                 + 0.99 b_k10_HSBC_s83 - 1.01 s_k10_HSBC_s83
                                  = 0
 w_k10_HSBC_s84_return_constraint: 1.0845 w_k10_HSBC_s8 - w_k10_HSBC_s84
                                 + 0.99 b_k10_HSBC_s84 - 1.01 s_k10_HSBC_s84
                                  = 0
 w_k10_HSBC_s85_return_constraint: 0.900772 w_k10_HSBC_s8 - w_k10_HSBC_s85
                                 + 0.99 b_k10_HSBC_s85 - 1.01 s_k10_HSBC_s85
                                  = 0
 w_k10_HSBC_s86_return_constraint: 0.812315 w_k10_HSBC_s8 - w_k10_HSBC_s86
                                 + 0.99 b_k10_HSBC_s86 - 1.01 s_k10_HSBC_s86
                                  = 0
 w_k10_HSBC_s87_return_constraint: 0.899172 w_k10_HSBC_s8 - w_k10_HSBC_s87
                                 + 0.99 b_k10_HSBC_s87 - 1.01 s_k10_HSBC_s87
                                  = 0
 w_k10_HSBC_s88_return_constraint: 0.845392 w_k10_HSBC_s8 - w_k10_HSBC_s88
                                 + 0.99 b_k10_HSBC_s88 - 1.01 s_k10_HSBC_s88
                                  = 0
 w_k10_HSBC_s89_return_constraint: 0.925896 w_k10_HSBC_s8 - w_k10_HSBC_s89
                                 + 0.99 b_k10_HSBC_s89 - 1.01 s_k10_HSBC_s89
                                  = 0
 w_k10_HSBC_s90_return_constraint: 1.16024 w_k10_HSBC_s9 - w_k10_HSBC_s90
                                 + 0.99 b_k10_HSBC_s90 - 1.01 s_k10_HSBC_s90
                                  = 0
 w_k10_HSBC_s91_return_constraint: 1.04407 w_k10_HSBC_s9 - w_k10_HSBC_s91
                                 + 0.99 b_k10_HSBC_s91 - 1.01 s_k10_HSBC_s91
                                  = 0
 w_k10_HSBC_s92_return_constraint: 1.21256 w_k10_HSBC_s9 - w_k10_HSBC_s92
                                 + 0.99 b_k10_HSBC_s92 - 1.01 s_k10_HSBC_s92
                                  = 0
 w_k10_HSBC_s93_return_constraint: 1.06106 w_k10_HSBC_s9 - w_k10_HSBC_s93
                                 + 0.99 b_k10_HSBC_s93 - 1.01 s_k10_HSBC_s93
                                  = 0
 w_k10_HSBC_s94_return_constraint: 1.02441 w_k10_HSBC_s9 - w_k10_HSBC_s94
                                 + 0.99 b_k10_HSBC_s94 - 1.01 s_k10_HSBC_s94
                                  = 0
 w_k10_HSBC_s95_return_constraint: 0.83947 w_k10_HSBC_s9 - w_k10_HSBC_s95
                                 + 0.99 b_k10_HSBC_s95 - 1.01 s_k10_HSBC_s95
                                  = 0
 w_k10_HSBC_s96_return_constraint: 1.00026 w_k10_HSBC_s9 - w_k10_HSBC_s96
                                 + 0.99 b_k10_HSBC_s96 - 1.01 s_k10_HSBC_s96
                                  = 0
 w_k10_HSBC_s97_return_constraint: 0.993911 w_k10_HSBC_s9 - w_k10_HSBC_s97
                                 + 0.99 b_k10_HSBC_s97 - 1.01 s_k10_HSBC_s97
                                  = 0
 w_k10_HSBC_s98_return_constraint: 0.953542 w_k10_HSBC_s9 - w_k10_HSBC_s98
                                 + 0.99 b_k10_HSBC_s98 - 1.01 s_k10_HSBC_s98
                                  = 0
 w_k10_HSBC_s99_return_constraint: 1.1464 w_k10_HSBC_s9 - w_k10_HSBC_s99
                                 + 0.99 b_k10_HSBC_s99 - 1.01 s_k10_HSBC_s99
                                  = 0
 w_k10_VOD_s0_return_constraint: 1.089 w_0_VOD - w_k10_VOD_s0
                                 + 0.99 b_k10_VOD_s0 - 1.01 s_k10_VOD_s0  = 0
 w_k10_VOD_s1_return_constraint: 1.00765 w_0_VOD - w_k10_VOD_s1
                                 + 0.99 b_k10_VOD_s1 - 1.01 s_k10_VOD_s1  = 0
 w_k10_VOD_s2_return_constraint: 1.04202 w_0_VOD - w_k10_VOD_s2
                                 + 0.99 b_k10_VOD_s2 - 1.01 s_k10_VOD_s2  = 0
 w_k10_VOD_s3_return_constraint: 1.05069 w_0_VOD - w_k10_VOD_s3
                                 + 0.99 b_k10_VOD_s3 - 1.01 s_k10_VOD_s3  = 0
 w_k10_VOD_s4_return_constraint: 0.965357 w_0_VOD - w_k10_VOD_s4
                                 + 0.99 b_k10_VOD_s4 - 1.01 s_k10_VOD_s4  = 0
 w_k10_VOD_s5_return_constraint: 0.906412 w_0_VOD - w_k10_VOD_s5
                                 + 0.99 b_k10_VOD_s5 - 1.01 s_k10_VOD_s5  = 0
 w_k10_VOD_s6_return_constraint: 0.92285 w_0_VOD - w_k10_VOD_s6
                                 + 0.99 b_k10_VOD_s6 - 1.01 s_k10_VOD_s6  = 0
 w_k10_VOD_s7_return_constraint: 1.04815 w_0_VOD - w_k10_VOD_s7
                                 + 0.99 b_k10_VOD_s7 - 1.01 s_k10_VOD_s7  = 0
 w_k10_VOD_s8_return_constraint: 0.919359 w_0_VOD - w_k10_VOD_s8
                                 + 0.99 b_k10_VOD_s8 - 1.01 s_k10_VOD_s8  = 0
 w_k10_VOD_s9_return_constraint: 0.907981 w_0_VOD - w_k10_VOD_s9
                                 + 0.99 b_k10_VOD_s9 - 1.01 s_k10_VOD_s9  = 0
 w_k10_VOD_s00_return_constraint: 0.984118 w_k10_VOD_s0 - w_k10_VOD_s00
                                 + 0.99 b_k10_VOD_s00 - 1.01 s_k10_VOD_s00  = 0
 w_k10_VOD_s01_return_constraint: 0.981695 w_k10_VOD_s0 - w_k10_VOD_s01
                                 + 0.99 b_k10_VOD_s01 - 1.01 s_k10_VOD_s01  = 0
 w_k10_VOD_s02_return_constraint: 0.955729 w_k10_VOD_s0 - w_k10_VOD_s02
                                 + 0.99 b_k10_VOD_s02 - 1.01 s_k10_VOD_s02  = 0
 w_k10_VOD_s03_return_constraint: 1.17235 w_k10_VOD_s0 - w_k10_VOD_s03
                                 + 0.99 b_k10_VOD_s03 - 1.01 s_k10_VOD_s03  = 0
 w_k10_VOD_s04_return_constraint: 1.0767 w_k10_VOD_s0 - w_k10_VOD_s04
                                 + 0.99 b_k10_VOD_s04 - 1.01 s_k10_VOD_s04  = 0
 w_k10_VOD_s05_return_constraint: 0.864046 w_k10_VOD_s0 - w_k10_VOD_s05
                                 + 0.99 b_k10_VOD_s05 - 1.01 s_k10_VOD_s05  = 0
 w_k10_VOD_s06_return_constraint: 1.12343 w_k10_VOD_s0 - w_k10_VOD_s06
                                 + 0.99 b_k10_VOD_s06 - 1.01 s_k10_VOD_s06  = 0
 w_k10_VOD_s07_return_constraint: 0.98208 w_k10_VOD_s0 - w_k10_VOD_s07
                                 + 0.99 b_k10_VOD_s07 - 1.01 s_k10_VOD_s07  = 0
 w_k10_VOD_s08_return_constraint: 1.04549 w_k10_VOD_s0 - w_k10_VOD_s08
                                 + 0.99 b_k10_VOD_s08 - 1.01 s_k10_VOD_s08  = 0
 w_k10_VOD_s09_return_constraint: 1.06603 w_k10_VOD_s0 - w_k10_VOD_s09
                                 + 0.99 b_k10_VOD_s09 - 1.01 s_k10_VOD_s09  = 0
 w_k10_VOD_s10_return_constraint: 0.947905 w_k10_VOD_s1 - w_k10_VOD_s10
                                 + 0.99 b_k10_VOD_s10 - 1.01 s_k10_VOD_s10  = 0
 w_k10_VOD_s11_return_constraint: 0.950877 w_k10_VOD_s1 - w_k10_VOD_s11
                                 + 0.99 b_k10_VOD_s11 - 1.01 s_k10_VOD_s11  = 0
 w_k10_VOD_s12_return_constraint: 1.069 w_k10_VOD_s1 - w_k10_VOD_s12
                                 + 0.99 b_k10_VOD_s12 - 1.01 s_k10_VOD_s12  = 0
 w_k10_VOD_s13_return_constraint: 0.826344 w_k10_VOD_s1 - w_k10_VOD_s13
                                 + 0.99 b_k10_VOD_s13 - 1.01 s_k10_VOD_s13  = 0
 w_k10_VOD_s14_return_constraint: 0.979988 w_k10_VOD_s1 - w_k10_VOD_s14
                                 + 0.99 b_k10_VOD_s14 - 1.01 s_k10_VOD_s14  = 0
 w_k10_VOD_s15_return_constraint: 1.11198 w_k10_VOD_s1 - w_k10_VOD_s15
                                 + 0.99 b_k10_VOD_s15 - 1.01 s_k10_VOD_s15  = 0
 w_k10_VOD_s16_return_constraint: 1.0537 w_k10_VOD_s1 - w_k10_VOD_s16
                                 + 0.99 b_k10_VOD_s16 - 1.01 s_k10_VOD_s16  = 0
 w_k10_VOD_s17_return_constraint: 0.920344 w_k10_VOD_s1 - w_k10_VOD_s17
                                 + 0.99 b_k10_VOD_s17 - 1.01 s_k10_VOD_s17  = 0
 w_k10_VOD_s18_return_constraint: 0.878652 w_k10_VOD_s1 - w_k10_VOD_s18
                                 + 0.99 b_k10_VOD_s18 - 1.01 s_k10_VOD_s18  = 0
 w_k10_VOD_s19_return_constraint: 1.06624 w_k10_VOD_s1 - w_k10_VOD_s19
                                 + 0.99 b_k10_VOD_s19 - 1.01 s_k10_VOD_s19  = 0
 w_k10_VOD_s20_return_constraint: 0.934795 w_k10_VOD_s2 - w_k10_VOD_s20
                                 + 0.99 b_k10_VOD_s20 - 1.01 s_k10_VOD_s20  = 0
 w_k10_VOD_s21_return_constraint: 0.975253 w_k10_VOD_s2 - w_k10_VOD_s21
                                 + 0.99 b_k10_VOD_s21 - 1.01 s_k10_VOD_s21  = 0
 w_k10_VOD_s22_return_constraint: 0.939604 w_k10_VOD_s2 - w_k10_VOD_s22
                                 + 0.99 b_k10_VOD_s22 - 1.01 s_k10_VOD_s22  = 0
 w_k10_VOD_s23_return_constraint: 1.09162 w_k10_VOD_s2 - w_k10_VOD_s23
                                 + 0.99 b_k10_VOD_s23 - 1.01 s_k10_VOD_s23  = 0
 w_k10_VOD_s24_return_constraint: 1.00133 w_k10_VOD_s2 - w_k10_VOD_s24
                                 + 0.99 b_k10_VOD_s24 - 1.01 s_k10_VOD_s24  = 0
 w_k10_VOD_s25_return_constraint: 0.980685 w_k10_VOD_s2 - w_k10_VOD_s25
                                 + 0.99 b_k10_VOD_s25 - 1.01 s_k10_VOD_s25  = 0
 w_k10_VOD_s26_return_constraint: 0.933213 w_k10_VOD_s2 - w_k10_VOD_s26
                                 + 0.99 b_k10_VOD_s26 - 1.01 s_k10_VOD_s26  = 0
 w_k10_VOD_s27_return_constraint: 0.92612 w_k10_VOD_s2 - w_k10_VOD_s27
                                 + 0.99 b_k10_VOD_s27 - 1.01 s_k10_VOD_s27  = 0
 w_k10_VOD_s28_return_constraint: 1.09044 w_k10_VOD_s2 - w_k10_VOD_s28
                                 + 0.99 b_k10_VOD_s28 - 1.01 s_k10_VOD_s28  = 0
 w_k10_VOD_s29_return_constraint: 1.01761 w_k10_VOD_s2 - w_k10_VOD_s29
                                 + 0.99 b_k10_VOD_s29 - 1.01 s_k10_VOD_s29  = 0
 w_k10_VOD_s30_return_constraint: 1.10232 w_k10_VOD_s3 - w_k10_VOD_s30
                                 + 0.99 b_k10_VOD_s30 - 1.01 s_k10_VOD_s30  = 0
 w_k10_VOD_s31_return_constraint: 1.0159 w_k10_VOD_s3 - w_k10_VOD_s31
                                 + 0.99 b_k10_VOD_s31 - 1.01 s_k10_VOD_s31  = 0
 w_k10_VOD_s32_return_constraint: 1.05016 w_k10_VOD_s3 - w_k10_VOD_s32
                                 + 0.99 b_k10_VOD_s32 - 1.01 s_k10_VOD_s32  = 0
 w_k10_VOD_s33_return_constraint: 1.05135 w_k10_VOD_s3 - w_k10_VOD_s33
                                 + 0.99 b_k10_VOD_s33 - 1.01 s_k10_VOD_s33  = 0
 w_k10_VOD_s34_return_constraint: 1.13144 w_k10_VOD_s3 - w_k10_VOD_s34
                                 + 0.99 b_k10_VOD_s34 - 1.01 s_k10_VOD_s34  = 0
 w_k10_VOD_s35_return_constraint: 1.02962 w_k10_VOD_s3 - w_k10_VOD_s35
                                 + 0.99 b_k10_VOD_s35 - 1.01 s_k10_VOD_s35  = 0
 w_k10_VOD_s36_return_constraint: 0.857618 w_k10_VOD_s3 - w_k10_VOD_s36
                                 + 0.99 b_k10_VOD_s36 - 1.01 s_k10_VOD_s36  = 0
 w_k10_VOD_s37_return_constraint: 0.958287 w_k10_VOD_s3 - w_k10_VOD_s37
                                 + 0.99 b_k10_VOD_s37 - 1.01 s_k10_VOD_s37  = 0
 w_k10_VOD_s38_return_constraint: 0.894504 w_k10_VOD_s3 - w_k10_VOD_s38
                                 + 0.99 b_k10_VOD_s38 - 1.01 s_k10_VOD_s38  = 0
 w_k10_VOD_s39_return_constraint: 0.914729 w_k10_VOD_s3 - w_k10_VOD_s39
                                 + 0.99 b_k10_VOD_s39 - 1.01 s_k10_VOD_s39  = 0
 w_k10_VOD_s40_return_constraint: 0.975529 w_k10_VOD_s4 - w_k10_VOD_s40
                                 + 0.99 b_k10_VOD_s40 - 1.01 s_k10_VOD_s40  = 0
 w_k10_VOD_s41_return_constraint: 0.970034 w_k10_VOD_s4 - w_k10_VOD_s41
                                 + 0.99 b_k10_VOD_s41 - 1.01 s_k10_VOD_s41  = 0
 w_k10_VOD_s42_return_constraint: 1.13543 w_k10_VOD_s4 - w_k10_VOD_s42
                                 + 0.99 b_k10_VOD_s42 - 1.01 s_k10_VOD_s42  = 0
 w_k10_VOD_s43_return_constraint: 0.923471 w_k10_VOD_s4 - w_k10_VOD_s43
                                 + 0.99 b_k10_VOD_s43 - 1.01 s_k10_VOD_s43  = 0
 w_k10_VOD_s44_return_constraint: 1.09344 w_k10_VOD_s4 - w_k10_VOD_s44
                                 + 0.99 b_k10_VOD_s44 - 1.01 s_k10_VOD_s44  = 0
 w_k10_VOD_s45_return_constraint: 1.10719 w_k10_VOD_s4 - w_k10_VOD_s45
                                 + 0.99 b_k10_VOD_s45 - 1.01 s_k10_VOD_s45  = 0
 w_k10_VOD_s46_return_constraint: 0.983577 w_k10_VOD_s4 - w_k10_VOD_s46
                                 + 0.99 b_k10_VOD_s46 - 1.01 s_k10_VOD_s46  = 0
 w_k10_VOD_s47_return_constraint: 1.22342 w_k10_VOD_s4 - w_k10_VOD_s47
                                 + 0.99 b_k10_VOD_s47 - 1.01 s_k10_VOD_s47  = 0
 w_k10_VOD_s48_return_constraint: 0.983718 w_k10_VOD_s4 - w_k10_VOD_s48
                                 + 0.99 b_k10_VOD_s48 - 1.01 s_k10_VOD_s48  = 0
 w_k10_VOD_s49_return_constraint: 0.933901 w_k10_VOD_s4 - w_k10_VOD_s49
                                 + 0.99 b_k10_VOD_s49 - 1.01 s_k10_VOD_s49  = 0
 w_k10_VOD_s50_return_constraint: 1.16521 w_k10_VOD_s5 - w_k10_VOD_s50
                                 + 0.99 b_k10_VOD_s50 - 1.01 s_k10_VOD_s50  = 0
 w_k10_VOD_s51_return_constraint: 1.06685 w_k10_VOD_s5 - w_k10_VOD_s51
                                 + 0.99 b_k10_VOD_s51 - 1.01 s_k10_VOD_s51  = 0
 w_k10_VOD_s52_return_constraint: 0.985015 w_k10_VOD_s5 - w_k10_VOD_s52
                                 + 0.99 b_k10_VOD_s52 - 1.01 s_k10_VOD_s52  = 0
 w_k10_VOD_s53_return_constraint: 1.05985 w_k10_VOD_s5 - w_k10_VOD_s53
                                 + 0.99 b_k10_VOD_s53 - 1.01 s_k10_VOD_s53  = 0
 w_k10_VOD_s54_return_constraint: 1.042 w_k10_VOD_s5 - w_k10_VOD_s54
                                 + 0.99 b_k10_VOD_s54 - 1.01 s_k10_VOD_s54  = 0
 w_k10_VOD_s55_return_constraint: 0.929178 w_k10_VOD_s5 - w_k10_VOD_s55
                                 + 0.99 b_k10_VOD_s55 - 1.01 s_k10_VOD_s55  = 0
 w_k10_VOD_s56_return_constraint: 1.05842 w_k10_VOD_s5 - w_k10_VOD_s56
                                 + 0.99 b_k10_VOD_s56 - 1.01 s_k10_VOD_s56  = 0
 w_k10_VOD_s57_return_constraint: 0.866359 w_k10_VOD_s5 - w_k10_VOD_s57
                                 + 0.99 b_k10_VOD_s57 - 1.01 s_k10_VOD_s57  = 0
 w_k10_VOD_s58_return_constraint: 0.934909 w_k10_VOD_s5 - w_k10_VOD_s58
                                 + 0.99 b_k10_VOD_s58 - 1.01 s_k10_VOD_s58  = 0
 w_k10_VOD_s59_return_constraint: 1.1162 w_k10_VOD_s5 - w_k10_VOD_s59
                                 + 0.99 b_k10_VOD_s59 - 1.01 s_k10_VOD_s59  = 0
 w_k10_VOD_s60_return_constraint: 0.99249 w_k10_VOD_s6 - w_k10_VOD_s60
                                 + 0.99 b_k10_VOD_s60 - 1.01 s_k10_VOD_s60  = 0
 w_k10_VOD_s61_return_constraint: 1.07899 w_k10_VOD_s6 - w_k10_VOD_s61
                                 + 0.99 b_k10_VOD_s61 - 1.01 s_k10_VOD_s61  = 0
 w_k10_VOD_s62_return_constraint: 1.03584 w_k10_VOD_s6 - w_k10_VOD_s62
                                 + 0.99 b_k10_VOD_s62 - 1.01 s_k10_VOD_s62  = 0
 w_k10_VOD_s63_return_constraint: 1.00539 w_k10_VOD_s6 - w_k10_VOD_s63
                                 + 0.99 b_k10_VOD_s63 - 1.01 s_k10_VOD_s63  = 0
 w_k10_VOD_s64_return_constraint: 0.884704 w_k10_VOD_s6 - w_k10_VOD_s64
                                 + 0.99 b_k10_VOD_s64 - 1.01 s_k10_VOD_s64  = 0
 w_k10_VOD_s65_return_constraint: 0.962849 w_k10_VOD_s6 - w_k10_VOD_s65
                                 + 0.99 b_k10_VOD_s65 - 1.01 s_k10_VOD_s65  = 0
 w_k10_VOD_s66_return_constraint: 1.11988 w_k10_VOD_s6 - w_k10_VOD_s66
                                 + 0.99 b_k10_VOD_s66 - 1.01 s_k10_VOD_s66  = 0
 w_k10_VOD_s67_return_constraint: 1.06062 w_k10_VOD_s6 - w_k10_VOD_s67
                                 + 0.99 b_k10_VOD_s67 - 1.01 s_k10_VOD_s67  = 0
 w_k10_VOD_s68_return_constraint: 1.17874 w_k10_VOD_s6 - w_k10_VOD_s68
                                 + 0.99 b_k10_VOD_s68 - 1.01 s_k10_VOD_s68  = 0
 w_k10_VOD_s69_return_constraint: 0.899098 w_k10_VOD_s6 - w_k10_VOD_s69
                                 + 0.99 b_k10_VOD_s69 - 1.01 s_k10_VOD_s69  = 0
 w_k10_VOD_s70_return_constraint: 1.08373 w_k10_VOD_s7 - w_k10_VOD_s70
                                 + 0.99 b_k10_VOD_s70 - 1.01 s_k10_VOD_s70  = 0
 w_k10_VOD_s71_return_constraint: 1.0389 w_k10_VOD_s7 - w_k10_VOD_s71
                                 + 0.99 b_k10_VOD_s71 - 1.01 s_k10_VOD_s71  = 0
 w_k10_VOD_s72_return_constraint: 1.05666 w_k10_VOD_s7 - w_k10_VOD_s72
                                 + 0.99 b_k10_VOD_s72 - 1.01 s_k10_VOD_s72  = 0
 w_k10_VOD_s73_return_constraint: 1.11695 w_k10_VOD_s7 - w_k10_VOD_s73
                                 + 0.99 b_k10_VOD_s73 - 1.01 s_k10_VOD_s73  = 0
 w_k10_VOD_s74_return_constraint: 0.94067 w_k10_VOD_s7 - w_k10_VOD_s74
                                 + 0.99 b_k10_VOD_s74 - 1.01 s_k10_VOD_s74  = 0
 w_k10_VOD_s75_return_constraint: 1.06672 w_k10_VOD_s7 - w_k10_VOD_s75
                                 + 0.99 b_k10_VOD_s75 - 1.01 s_k10_VOD_s75  = 0
 w_k10_VOD_s76_return_constraint: 0.9479 w_k10_VOD_s7 - w_k10_VOD_s76
                                 + 0.99 b_k10_VOD_s76 - 1.01 s_k10_VOD_s76  = 0
 w_k10_VOD_s77_return_constraint: 1.19981 w_k10_VOD_s7 - w_k10_VOD_s77
                                 + 0.99 b_k10_VOD_s77 - 1.01 s_k10_VOD_s77  = 0
 w_k10_VOD_s78_return_constraint: 0.93358 w_k10_VOD_s7 - w_k10_VOD_s78
                                 + 0.99 b_k10_VOD_s78 - 1.01 s_k10_VOD_s78  = 0
 w_k10_VOD_s79_return_constraint: 1.10968 w_k10_VOD_s7 - w_k10_VOD_s79
                                 + 0.99 b_k10_VOD_s79 - 1.01 s_k10_VOD_s79  = 0
 w_k10_VOD_s80_return_constraint: 1.08626 w_k10_VOD_s8 - w_k10_VOD_s80
                                 + 0.99 b_k10_VOD_s80 - 1.01 s_k10_VOD_s80  = 0
 w_k10_VOD_s81_return_constraint: 1.02161 w_k10_VOD_s8 - w_k10_VOD_s81
                                 + 0.99 b_k10_VOD_s81 - 1.01 s_k10_VOD_s81  = 0
 w_k10_VOD_s82_return_constraint: 1.13691 w_k10_VOD_s8 - w_k10_VOD_s82
                                 + 0.99 b_k10_VOD_s82 - 1.01 s_k10_VOD_s82  = 0
 w_k10_VOD_s83_return_constraint: 1.12306 w_k10_VOD_s8 - w_k10_VOD_s83
                                 + 0.99 b_k10_VOD_s83 - 1.01 s_k10_VOD_s83  = 0
 w_k10_VOD_s84_return_constraint: 1.03016 w_k10_VOD_s8 - w_k10_VOD_s84
                                 + 0.99 b_k10_VOD_s84 - 1.01 s_k10_VOD_s84  = 0
 w_k10_VOD_s85_return_constraint: 0.955794 w_k10_VOD_s8 - w_k10_VOD_s85
                                 + 0.99 b_k10_VOD_s85 - 1.01 s_k10_VOD_s85  = 0
 w_k10_VOD_s86_return_constraint: 0.979806 w_k10_VOD_s8 - w_k10_VOD_s86
                                 + 0.99 b_k10_VOD_s86 - 1.01 s_k10_VOD_s86  = 0
 w_k10_VOD_s87_return_constraint: 1.0823 w_k10_VOD_s8 - w_k10_VOD_s87
                                 + 0.99 b_k10_VOD_s87 - 1.01 s_k10_VOD_s87  = 0
 w_k10_VOD_s88_return_constraint: 1.04676 w_k10_VOD_s8 - w_k10_VOD_s88
                                 + 0.99 b_k10_VOD_s88 - 1.01 s_k10_VOD_s88  = 0
 w_k10_VOD_s89_return_constraint: 0.968458 w_k10_VOD_s8 - w_k10_VOD_s89
                                 + 0.99 b_k10_VOD_s89 - 1.01 s_k10_VOD_s89  = 0
 w_k10_VOD_s90_return_constraint: 1.13919 w_k10_VOD_s9 - w_k10_VOD_s90
                                 + 0.99 b_k10_VOD_s90 - 1.01 s_k10_VOD_s90  = 0
 w_k10_VOD_s91_return_constraint: 1.04488 w_k10_VOD_s9 - w_k10_VOD_s91
                                 + 0.99 b_k10_VOD_s91 - 1.01 s_k10_VOD_s91  = 0
 w_k10_VOD_s92_return_constraint: 1.12584 w_k10_VOD_s9 - w_k10_VOD_s92
                                 + 0.99 b_k10_VOD_s92 - 1.01 s_k10_VOD_s92  = 0
 w_k10_VOD_s93_return_constraint: 1.02575 w_k10_VOD_s9 - w_k10_VOD_s93
                                 + 0.99 b_k10_VOD_s93 - 1.01 s_k10_VOD_s93  = 0
 w_k10_VOD_s94_return_constraint: 1.08357 w_k10_VOD_s9 - w_k10_VOD_s94
                                 + 0.99 b_k10_VOD_s94 - 1.01 s_k10_VOD_s94  = 0
 w_k10_VOD_s95_return_constraint: 1.01665 w_k10_VOD_s9 - w_k10_VOD_s95
                                 + 0.99 b_k10_VOD_s95 - 1.01 s_k10_VOD_s95  = 0
 w_k10_VOD_s96_return_constraint: 0.957789 w_k10_VOD_s9 - w_k10_VOD_s96
                                 + 0.99 b_k10_VOD_s96 - 1.01 s_k10_VOD_s96  = 0
 w_k10_VOD_s97_return_constraint: 1.00785 w_k10_VOD_s9 - w_k10_VOD_s97
                                 + 0.99 b_k10_VOD_s97 - 1.01 s_k10_VOD_s97  = 0
 w_k10_VOD_s98_return_constraint: 1.00513 w_k10_VOD_s9 - w_k10_VOD_s98
                                 + 0.99 b_k10_VOD_s98 - 1.01 s_k10_VOD_s98  = 0
 w_k10_VOD_s99_return_constraint: 1.10902 w_k10_VOD_s9 - w_k10_VOD_s99
                                 + 0.99 b_k10_VOD_s99 - 1.01 s_k10_VOD_s99  = 0
 w_k10_BP_s0_return_constraint:  1.21928 w_0_BP - w_k10_BP_s0
                                 + 0.99 b_k10_BP_s0 - 1.01 s_k10_BP_s0  = 0
 w_k10_BP_s1_return_constraint:  0.853714 w_0_BP - w_k10_BP_s1
                                 + 0.99 b_k10_BP_s1 - 1.01 s_k10_BP_s1  = 0
 w_k10_BP_s2_return_constraint:  1.13971 w_0_BP - w_k10_BP_s2
                                 + 0.99 b_k10_BP_s2 - 1.01 s_k10_BP_s2  = 0
 w_k10_BP_s3_return_constraint:  0.995312 w_0_BP - w_k10_BP_s3
                                 + 0.99 b_k10_BP_s3 - 1.01 s_k10_BP_s3  = 0
 w_k10_BP_s4_return_constraint:  0.986295 w_0_BP - w_k10_BP_s4
                                 + 0.99 b_k10_BP_s4 - 1.01 s_k10_BP_s4  = 0
 w_k10_BP_s5_return_constraint:  0.789553 w_0_BP - w_k10_BP_s5
                                 + 0.99 b_k10_BP_s5 - 1.01 s_k10_BP_s5  = 0
 w_k10_BP_s6_return_constraint:  0.860596 w_0_BP - w_k10_BP_s6
                                 + 0.99 b_k10_BP_s6 - 1.01 s_k10_BP_s6  = 0
 w_k10_BP_s7_return_constraint:  1.09671 w_0_BP - w_k10_BP_s7
                                 + 0.99 b_k10_BP_s7 - 1.01 s_k10_BP_s7  = 0
 w_k10_BP_s8_return_constraint:  0.977778 w_0_BP - w_k10_BP_s8
                                 + 0.99 b_k10_BP_s8 - 1.01 s_k10_BP_s8  = 0
 w_k10_BP_s9_return_constraint:  0.834144 w_0_BP - w_k10_BP_s9
                                 + 0.99 b_k10_BP_s9 - 1.01 s_k10_BP_s9  = 0
 w_k10_BP_s00_return_constraint: 0.934296 w_k10_BP_s0 - w_k10_BP_s00
                                 + 0.99 b_k10_BP_s00 - 1.01 s_k10_BP_s00  = 0
 w_k10_BP_s01_return_constraint: 1.26861 w_k10_BP_s0 - w_k10_BP_s01
                                 + 0.99 b_k10_BP_s01 - 1.01 s_k10_BP_s01  = 0
 w_k10_BP_s02_return_constraint: 1.07028 w_k10_BP_s0 - w_k10_BP_s02
                                 + 0.99 b_k10_BP_s02 - 1.01 s_k10_BP_s02  = 0
 w_k10_BP_s03_return_constraint: 1.46819 w_k10_BP_s0 - w_k10_BP_s03
                                 + 0.99 b_k10_BP_s03 - 1.01 s_k10_BP_s03  = 0
 w_k10_BP_s04_return_constraint: 0.916933 w_k10_BP_s0 - w_k10_BP_s04
                                 + 0.99 b_k10_BP_s04 - 1.01 s_k10_BP_s04  = 0
 w_k10_BP_s05_return_constraint: 0.708845 w_k10_BP_s0 - w_k10_BP_s05
                                 + 0.99 b_k10_BP_s05 - 1.01 s_k10_BP_s05  = 0
 w_k10_BP_s06_return_constraint: 1.03088 w_k10_BP_s0 - w_k10_BP_s06
                                 + 0.99 b_k10_BP_s06 - 1.01 s_k10_BP_s06  = 0
 w_k10_BP_s07_return_constraint: 0.867441 w_k10_BP_s0 - w_k10_BP_s07
                                 + 0.99 b_k10_BP_s07 - 1.01 s_k10_BP_s07  = 0
 w_k10_BP_s08_return_constraint: 1.12298 w_k10_BP_s0 - w_k10_BP_s08
                                 + 0.99 b_k10_BP_s08 - 1.01 s_k10_BP_s08  = 0
 w_k10_BP_s09_return_constraint: 1.20033 w_k10_BP_s0 - w_k10_BP_s09
                                 + 0.99 b_k10_BP_s09 - 1.01 s_k10_BP_s09  = 0
 w_k10_BP_s10_return_constraint: 1.11766 w_k10_BP_s1 - w_k10_BP_s10
                                 + 0.99 b_k10_BP_s10 - 1.01 s_k10_BP_s10  = 0
 w_k10_BP_s11_return_constraint: 0.855042 w_k10_BP_s1 - w_k10_BP_s11
                                 + 0.99 b_k10_BP_s11 - 1.01 s_k10_BP_s11  = 0
 w_k10_BP_s12_return_constraint: 1.17937 w_k10_BP_s1 - w_k10_BP_s12
                                 + 0.99 b_k10_BP_s12 - 1.01 s_k10_BP_s12  = 0
 w_k10_BP_s13_return_constraint: 0.723858 w_k10_BP_s1 - w_k10_BP_s13
                                 + 0.99 b_k10_BP_s13 - 1.01 s_k10_BP_s13  = 0
 w_k10_BP_s14_return_constraint: 0.925614 w_k10_BP_s1 - w_k10_BP_s14
                                 + 0.99 b_k10_BP_s14 - 1.01 s_k10_BP_s14  = 0
 w_k10_BP_s15_return_constraint: 1.32565 w_k10_BP_s1 - w_k10_BP_s15
                                 + 0.99 b_k10_BP_s15 - 1.01 s_k10_BP_s15  = 0
 w_k10_BP_s16_return_constraint: 1.00002 w_k10_BP_s1 - w_k10_BP_s16
                                 + 0.99 b_k10_BP_s16 - 1.01 s_k10_BP_s16  = 0
 w_k10_BP_s17_return_constraint: 1.03941 w_k10_BP_s1 - w_k10_BP_s17
                                 + 0.99 b_k10_BP_s17 - 1.01 s_k10_BP_s17  = 0
 w_k10_BP_s18_return_constraint: 0.77358 w_k10_BP_s1 - w_k10_BP_s18
                                 + 0.99 b_k10_BP_s18 - 1.01 s_k10_BP_s18  = 0
 w_k10_BP_s19_return_constraint: 0.948694 w_k10_BP_s1 - w_k10_BP_s19
                                 + 0.99 b_k10_BP_s19 - 1.01 s_k10_BP_s19  = 0
 w_k10_BP_s20_return_constraint: 0.779914 w_k10_BP_s2 - w_k10_BP_s20
                                 + 0.99 b_k10_BP_s20 - 1.01 s_k10_BP_s20  = 0
 w_k10_BP_s21_return_constraint: 1.01921 w_k10_BP_s2 - w_k10_BP_s21
                                 + 0.99 b_k10_BP_s21 - 1.01 s_k10_BP_s21  = 0
 w_k10_BP_s22_return_constraint: 0.941681 w_k10_BP_s2 - w_k10_BP_s22
                                 + 0.99 b_k10_BP_s22 - 1.01 s_k10_BP_s22  = 0
 w_k10_BP_s23_return_constraint: 1.25563 w_k10_BP_s2 - w_k10_BP_s23
                                 + 0.99 b_k10_BP_s23 - 1.01 s_k10_BP_s23  = 0
 w_k10_BP_s24_return_constraint: 1.03847 w_k10_BP_s2 - w_k10_BP_s24
                                 + 0.99 b_k10_BP_s24 - 1.01 s_k10_BP_s24  = 0
 w_k10_BP_s25_return_constraint: 0.928697 w_k10_BP_s2 - w_k10_BP_s25
                                 + 0.99 b_k10_BP_s25 - 1.01 s_k10_BP_s25  = 0
 w_k10_BP_s26_return_constraint: 0.870007 w_k10_BP_s2 - w_k10_BP_s26
                                 + 0.99 b_k10_BP_s26 - 1.01 s_k10_BP_s26  = 0
 w_k10_BP_s27_return_constraint: 1.07808 w_k10_BP_s2 - w_k10_BP_s27
                                 + 0.99 b_k10_BP_s27 - 1.01 s_k10_BP_s27  = 0
 w_k10_BP_s28_return_constraint: 1.42011 w_k10_BP_s2 - w_k10_BP_s28
                                 + 0.99 b_k10_BP_s28 - 1.01 s_k10_BP_s28  = 0
 w_k10_BP_s29_return_constraint: 1.12074 w_k10_BP_s2 - w_k10_BP_s29
                                 + 0.99 b_k10_BP_s29 - 1.01 s_k10_BP_s29  = 0
 w_k10_BP_s30_return_constraint: 1.17676 w_k10_BP_s3 - w_k10_BP_s30
                                 + 0.99 b_k10_BP_s30 - 1.01 s_k10_BP_s30  = 0
 w_k10_BP_s31_return_constraint: 0.942823 w_k10_BP_s3 - w_k10_BP_s31
                                 + 0.99 b_k10_BP_s31 - 1.01 s_k10_BP_s31  = 0
 w_k10_BP_s32_return_constraint: 1.08851 w_k10_BP_s3 - w_k10_BP_s32
                                 + 0.99 b_k10_BP_s32 - 1.01 s_k10_BP_s32  = 0
 w_k10_BP_s33_return_constraint: 0.904927 w_k10_BP_s3 - w_k10_BP_s33
                                 + 0.99 b_k10_BP_s33 - 1.01 s_k10_BP_s33  = 0
 w_k10_BP_s34_return_constraint: 1.16205 w_k10_BP_s3 - w_k10_BP_s34
                                 + 0.99 b_k10_BP_s34 - 1.01 s_k10_BP_s34  = 0
 w_k10_BP_s35_return_constraint: 1.28168 w_k10_BP_s3 - w_k10_BP_s35
                                 + 0.99 b_k10_BP_s35 - 1.01 s_k10_BP_s35  = 0
 w_k10_BP_s36_return_constraint: 0.868624 w_k10_BP_s3 - w_k10_BP_s36
                                 + 0.99 b_k10_BP_s36 - 1.01 s_k10_BP_s36  = 0
 w_k10_BP_s37_return_constraint: 0.96806 w_k10_BP_s3 - w_k10_BP_s37
                                 + 0.99 b_k10_BP_s37 - 1.01 s_k10_BP_s37  = 0
 w_k10_BP_s38_return_constraint: 1.19553 w_k10_BP_s3 - w_k10_BP_s38
                                 + 0.99 b_k10_BP_s38 - 1.01 s_k10_BP_s38  = 0
 w_k10_BP_s39_return_constraint: 0.80664 w_k10_BP_s3 - w_k10_BP_s39
                                 + 0.99 b_k10_BP_s39 - 1.01 s_k10_BP_s39  = 0
 w_k10_BP_s40_return_constraint: 0.841677 w_k10_BP_s4 - w_k10_BP_s40
                                 + 0.99 b_k10_BP_s40 - 1.01 s_k10_BP_s40  = 0
 w_k10_BP_s41_return_constraint: 0.948035 w_k10_BP_s4 - w_k10_BP_s41
                                 + 0.99 b_k10_BP_s41 - 1.01 s_k10_BP_s41  = 0
 w_k10_BP_s42_return_constraint: 1.30651 w_k10_BP_s4 - w_k10_BP_s42
                                 + 0.99 b_k10_BP_s42 - 1.01 s_k10_BP_s42  = 0
 w_k10_BP_s43_return_constraint: 0.937058 w_k10_BP_s4 - w_k10_BP_s43
                                 + 0.99 b_k10_BP_s43 - 1.01 s_k10_BP_s43  = 0
 w_k10_BP_s44_return_constraint: 1.03109 w_k10_BP_s4 - w_k10_BP_s44
                                 + 0.99 b_k10_BP_s44 - 1.01 s_k10_BP_s44  = 0
 w_k10_BP_s45_return_constraint: 0.940658 w_k10_BP_s4 - w_k10_BP_s45
                                 + 0.99 b_k10_BP_s45 - 1.01 s_k10_BP_s45  = 0
 w_k10_BP_s46_return_constraint: 1.10437 w_k10_BP_s4 - w_k10_BP_s46
                                 + 0.99 b_k10_BP_s46 - 1.01 s_k10_BP_s46  = 0
 w_k10_BP_s47_return_constraint: 1.19796 w_k10_BP_s4 - w_k10_BP_s47
                                 + 0.99 b_k10_BP_s47 - 1.01 s_k10_BP_s47  = 0
 w_k10_BP_s48_return_constraint: 1.05455 w_k10_BP_s4 - w_k10_BP_s48
                                 + 0.99 b_k10_BP_s48 - 1.01 s_k10_BP_s48  = 0
 w_k10_BP_s49_return_constraint: 0.725063 w_k10_BP_s4 - w_k10_BP_s49
                                 + 0.99 b_k10_BP_s49 - 1.01 s_k10_BP_s49  = 0
 w_k10_BP_s50_return_constraint: 1.33139 w_k10_BP_s5 - w_k10_BP_s50
                                 + 0.99 b_k10_BP_s50 - 1.01 s_k10_BP_s50  = 0
 w_k10_BP_s51_return_constraint: 0.906453 w_k10_BP_s5 - w_k10_BP_s51
                                 + 0.99 b_k10_BP_s51 - 1.01 s_k10_BP_s51  = 0
 w_k10_BP_s52_return_constraint: 0.976057 w_k10_BP_s5 - w_k10_BP_s52
                                 + 0.99 b_k10_BP_s52 - 1.01 s_k10_BP_s52  = 0
 w_k10_BP_s53_return_constraint: 1.09833 w_k10_BP_s5 - w_k10_BP_s53
                                 + 0.99 b_k10_BP_s53 - 1.01 s_k10_BP_s53  = 0
 w_k10_BP_s54_return_constraint: 0.945161 w_k10_BP_s5 - w_k10_BP_s54
                                 + 0.99 b_k10_BP_s54 - 1.01 s_k10_BP_s54  = 0
 w_k10_BP_s55_return_constraint: 0.794375 w_k10_BP_s5 - w_k10_BP_s55
                                 + 0.99 b_k10_BP_s55 - 1.01 s_k10_BP_s55  = 0
 w_k10_BP_s56_return_constraint: 1.11357 w_k10_BP_s5 - w_k10_BP_s56
                                 + 0.99 b_k10_BP_s56 - 1.01 s_k10_BP_s56  = 0
 w_k10_BP_s57_return_constraint: 1.02865 w_k10_BP_s5 - w_k10_BP_s57
                                 + 0.99 b_k10_BP_s57 - 1.01 s_k10_BP_s57  = 0
 w_k10_BP_s58_return_constraint: 0.97491 w_k10_BP_s5 - w_k10_BP_s58
                                 + 0.99 b_k10_BP_s58 - 1.01 s_k10_BP_s58  = 0
 w_k10_BP_s59_return_constraint: 1.11569 w_k10_BP_s5 - w_k10_BP_s59
                                 + 0.99 b_k10_BP_s59 - 1.01 s_k10_BP_s59  = 0
 w_k10_BP_s60_return_constraint: 0.840676 w_k10_BP_s6 - w_k10_BP_s60
                                 + 0.99 b_k10_BP_s60 - 1.01 s_k10_BP_s60  = 0
 w_k10_BP_s61_return_constraint: 1.26219 w_k10_BP_s6 - w_k10_BP_s61
                                 + 0.99 b_k10_BP_s61 - 1.01 s_k10_BP_s61  = 0
 w_k10_BP_s62_return_constraint: 0.981941 w_k10_BP_s6 - w_k10_BP_s62
                                 + 0.99 b_k10_BP_s62 - 1.01 s_k10_BP_s62  = 0
 w_k10_BP_s63_return_constraint: 0.970966 w_k10_BP_s6 - w_k10_BP_s63
                                 + 0.99 b_k10_BP_s63 - 1.01 s_k10_BP_s63  = 0
 w_k10_BP_s64_return_constraint: 0.77543 w_k10_BP_s6 - w_k10_BP_s64
                                 + 0.99 b_k10_BP_s64 - 1.01 s_k10_BP_s64  = 0
 w_k10_BP_s65_return_constraint: 0.86337 w_k10_BP_s6 - w_k10_BP_s65
                                 + 0.99 b_k10_BP_s65 - 1.01 s_k10_BP_s65  = 0
 w_k10_BP_s66_return_constraint: 1.0778 w_k10_BP_s6 - w_k10_BP_s66
                                 + 0.99 b_k10_BP_s66 - 1.01 s_k10_BP_s66  = 0
 w_k10_BP_s67_return_constraint: 1.11693 w_k10_BP_s6 - w_k10_BP_s67
                                 + 0.99 b_k10_BP_s67 - 1.01 s_k10_BP_s67  = 0
 w_k10_BP_s68_return_constraint: 1.22841 w_k10_BP_s6 - w_k10_BP_s68
                                 + 0.99 b_k10_BP_s68 - 1.01 s_k10_BP_s68  = 0
 w_k10_BP_s69_return_constraint: 0.972545 w_k10_BP_s6 - w_k10_BP_s69
                                 + 0.99 b_k10_BP_s69 - 1.01 s_k10_BP_s69  = 0
 w_k10_BP_s70_return_constraint: 1.03798 w_k10_BP_s7 - w_k10_BP_s70
                                 + 0.99 b_k10_BP_s70 - 1.01 s_k10_BP_s70  = 0
 w_k10_BP_s71_return_constraint: 1.13738 w_k10_BP_s7 - w_k10_BP_s71
                                 + 0.99 b_k10_BP_s71 - 1.01 s_k10_BP_s71  = 0
 w_k10_BP_s72_return_constraint: 1.04119 w_k10_BP_s7 - w_k10_BP_s72
                                 + 0.99 b_k10_BP_s72 - 1.01 s_k10_BP_s72  = 0
 w_k10_BP_s73_return_constraint: 1.25396 w_k10_BP_s7 - w_k10_BP_s73
                                 + 0.99 b_k10_BP_s73 - 1.01 s_k10_BP_s73  = 0
 w_k10_BP_s74_return_constraint: 0.880787 w_k10_BP_s7 - w_k10_BP_s74
                                 + 0.99 b_k10_BP_s74 - 1.01 s_k10_BP_s74  = 0
 w_k10_BP_s75_return_constraint: 1.00992 w_k10_BP_s7 - w_k10_BP_s75
                                 + 0.99 b_k10_BP_s75 - 1.01 s_k10_BP_s75  = 0
 w_k10_BP_s76_return_constraint: 0.873136 w_k10_BP_s7 - w_k10_BP_s76
                                 + 0.99 b_k10_BP_s76 - 1.01 s_k10_BP_s76  = 0
 w_k10_BP_s77_return_constraint: 1.30525 w_k10_BP_s7 - w_k10_BP_s77
                                 + 0.99 b_k10_BP_s77 - 1.01 s_k10_BP_s77  = 0
 w_k10_BP_s78_return_constraint: 0.868558 w_k10_BP_s7 - w_k10_BP_s78
                                 + 0.99 b_k10_BP_s78 - 1.01 s_k10_BP_s78  = 0
 w_k10_BP_s79_return_constraint: 1.26375 w_k10_BP_s7 - w_k10_BP_s79
                                 + 0.99 b_k10_BP_s79 - 1.01 s_k10_BP_s79  = 0
 w_k10_BP_s80_return_constraint: 1.11414 w_k10_BP_s8 - w_k10_BP_s80
                                 + 0.99 b_k10_BP_s80 - 1.01 s_k10_BP_s80  = 0
 w_k10_BP_s81_return_constraint: 0.962344 w_k10_BP_s8 - w_k10_BP_s81
                                 + 0.99 b_k10_BP_s81 - 1.01 s_k10_BP_s81  = 0
 w_k10_BP_s82_return_constraint: 1.18781 w_k10_BP_s8 - w_k10_BP_s82
                                 + 0.99 b_k10_BP_s82 - 1.01 s_k10_BP_s82  = 0
 w_k10_BP_s83_return_constraint: 1.38164 w_k10_BP_s8 - w_k10_BP_s83
                                 + 0.99 b_k10_BP_s83 - 1.01 s_k10_BP_s83  = 0
 w_k10_BP_s84_return_constraint: 1.08341 w_k10_BP_s8 - w_k10_BP_s84
                                 + 0.99 b_k10_BP_s84 - 1.01 s_k10_BP_s84  = 0
 w_k10_BP_s85_return_constraint: 0.856839 w_k10_BP_s8 - w_k10_BP_s85
                                 + 0.99 b_k10_BP_s85 - 1.01 s_k10_BP_s85  = 0
 w_k10_BP_s86_return_constraint: 0.85754 w_k10_BP_s8 - w_k10_BP_s86
                                 + 0.99 b_k10_BP_s86 - 1.01 s_k10_BP_s86  = 0
 w_k10_BP_s87_return_constraint: 1.14696 w_k10_BP_s8 - w_k10_BP_s87
                                 + 0.99 b_k10_BP_s87 - 1.01 s_k10_BP_s87  = 0
 w_k10_BP_s88_return_constraint: 0.961497 w_k10_BP_s8 - w_k10_BP_s88
                                 + 0.99 b_k10_BP_s88 - 1.01 s_k10_BP_s88  = 0
 w_k10_BP_s89_return_constraint: 0.772754 w_k10_BP_s8 - w_k10_BP_s89
                                 + 0.99 b_k10_BP_s89 - 1.01 s_k10_BP_s89  = 0
 w_k10_BP_s90_return_constraint: 1.24812 w_k10_BP_s9 - w_k10_BP_s90
                                 + 0.99 b_k10_BP_s90 - 1.01 s_k10_BP_s90  = 0
 w_k10_BP_s91_return_constraint: 1.04654 w_k10_BP_s9 - w_k10_BP_s91
                                 + 0.99 b_k10_BP_s91 - 1.01 s_k10_BP_s91  = 0
 w_k10_BP_s92_return_constraint: 1.32905 w_k10_BP_s9 - w_k10_BP_s92
                                 + 0.99 b_k10_BP_s92 - 1.01 s_k10_BP_s92  = 0
 w_k10_BP_s93_return_constraint: 1.0717 w_k10_BP_s9 - w_k10_BP_s93
                                 + 0.99 b_k10_BP_s93 - 1.01 s_k10_BP_s93  = 0
 w_k10_BP_s94_return_constraint: 1.04467 w_k10_BP_s9 - w_k10_BP_s94
                                 + 0.99 b_k10_BP_s94 - 1.01 s_k10_BP_s94  = 0
 w_k10_BP_s95_return_constraint: 0.743343 w_k10_BP_s9 - w_k10_BP_s95
                                 + 0.99 b_k10_BP_s95 - 1.01 s_k10_BP_s95  = 0
 w_k10_BP_s96_return_constraint: 0.891225 w_k10_BP_s9 - w_k10_BP_s96
                                 + 0.99 b_k10_BP_s96 - 1.01 s_k10_BP_s96  = 0
 w_k10_BP_s97_return_constraint: 0.981274 w_k10_BP_s9 - w_k10_BP_s97
                                 + 0.99 b_k10_BP_s97 - 1.01 s_k10_BP_s97  = 0
 w_k10_BP_s98_return_constraint: 0.878973 w_k10_BP_s9 - w_k10_BP_s98
                                 + 0.99 b_k10_BP_s98 - 1.01 s_k10_BP_s98  = 0
 w_k10_BP_s99_return_constraint: 1.18664 w_k10_BP_s9 - w_k10_BP_s99
                                 + 0.99 b_k10_BP_s99 - 1.01 s_k10_BP_s99  = 0
 w_k10_GSK_s0_return_constraint: 1.1222 w_0_GSK - w_k10_GSK_s0
                                 + 0.99 b_k10_GSK_s0 - 1.01 s_k10_GSK_s0  = 0
 w_k10_GSK_s1_return_constraint: 0.920516 w_0_GSK - w_k10_GSK_s1
                                 + 0.99 b_k10_GSK_s1 - 1.01 s_k10_GSK_s1  = 0
 w_k10_GSK_s2_return_constraint: 1.08505 w_0_GSK - w_k10_GSK_s2
                                 + 0.99 b_k10_GSK_s2 - 1.01 s_k10_GSK_s2  = 0
 w_k10_GSK_s3_return_constraint: 0.997947 w_0_GSK - w_k10_GSK_s3
                                 + 0.99 b_k10_GSK_s3 - 1.01 s_k10_GSK_s3  = 0
 w_k10_GSK_s4_return_constraint: 0.999151 w_0_GSK - w_k10_GSK_s4
                                 + 0.99 b_k10_GSK_s4 - 1.01 s_k10_GSK_s4  = 0
 w_k10_GSK_s5_return_constraint: 0.923488 w_0_GSK - w_k10_GSK_s5
                                 + 0.99 b_k10_GSK_s5 - 1.01 s_k10_GSK_s5  = 0
 w_k10_GSK_s6_return_constraint: 0.862114 w_0_GSK - w_k10_GSK_s6
                                 + 0.99 b_k10_GSK_s6 - 1.01 s_k10_GSK_s6  = 0
 w_k10_GSK_s7_return_constraint: 1.07199 w_0_GSK - w_k10_GSK_s7
                                 + 0.99 b_k10_GSK_s7 - 1.01 s_k10_GSK_s7  = 0
 w_k10_GSK_s8_return_constraint: 0.989139 w_0_GSK - w_k10_GSK_s8
                                 + 0.99 b_k10_GSK_s8 - 1.01 s_k10_GSK_s8  = 0
 w_k10_GSK_s9_return_constraint: 0.918067 w_0_GSK - w_k10_GSK_s9
                                 + 0.99 b_k10_GSK_s9 - 1.01 s_k10_GSK_s9  = 0
 w_k10_GSK_s00_return_constraint: 0.975828 w_k10_GSK_s0 - w_k10_GSK_s00
                                 + 0.99 b_k10_GSK_s00 - 1.01 s_k10_GSK_s00  = 0
 w_k10_GSK_s01_return_constraint: 1.15599 w_k10_GSK_s0 - w_k10_GSK_s01
                                 + 0.99 b_k10_GSK_s01 - 1.01 s_k10_GSK_s01  = 0
 w_k10_GSK_s02_return_constraint: 1.09133 w_k10_GSK_s0 - w_k10_GSK_s02
                                 + 0.99 b_k10_GSK_s02 - 1.01 s_k10_GSK_s02  = 0
 w_k10_GSK_s03_return_constraint: 1.29743 w_k10_GSK_s0 - w_k10_GSK_s03
                                 + 0.99 b_k10_GSK_s03 - 1.01 s_k10_GSK_s03  = 0
 w_k10_GSK_s04_return_constraint: 1.06894 w_k10_GSK_s0 - w_k10_GSK_s04
                                 + 0.99 b_k10_GSK_s04 - 1.01 s_k10_GSK_s04  = 0
 w_k10_GSK_s05_return_constraint: 0.79082 w_k10_GSK_s0 - w_k10_GSK_s05
                                 + 0.99 b_k10_GSK_s05 - 1.01 s_k10_GSK_s05  = 0
 w_k10_GSK_s06_return_constraint: 1.01122 w_k10_GSK_s0 - w_k10_GSK_s06
                                 + 0.99 b_k10_GSK_s06 - 1.01 s_k10_GSK_s06  = 0
 w_k10_GSK_s07_return_constraint: 0.943978 w_k10_GSK_s0 - w_k10_GSK_s07
                                 + 0.99 b_k10_GSK_s07 - 1.01 s_k10_GSK_s07  = 0
 w_k10_GSK_s08_return_constraint: 0.977452 w_k10_GSK_s0 - w_k10_GSK_s08
                                 + 0.99 b_k10_GSK_s08 - 1.01 s_k10_GSK_s08  = 0
 w_k10_GSK_s09_return_constraint: 1.0977 w_k10_GSK_s0 - w_k10_GSK_s09
                                 + 0.99 b_k10_GSK_s09 - 1.01 s_k10_GSK_s09  = 0
 w_k10_GSK_s10_return_constraint: 1.08034 w_k10_GSK_s1 - w_k10_GSK_s10
                                 + 0.99 b_k10_GSK_s10 - 1.01 s_k10_GSK_s10  = 0
 w_k10_GSK_s11_return_constraint: 0.84636 w_k10_GSK_s1 - w_k10_GSK_s11
                                 + 0.99 b_k10_GSK_s11 - 1.01 s_k10_GSK_s11  = 0
 w_k10_GSK_s12_return_constraint: 1.04233 w_k10_GSK_s1 - w_k10_GSK_s12
                                 + 0.99 b_k10_GSK_s12 - 1.01 s_k10_GSK_s12  = 0
 w_k10_GSK_s13_return_constraint: 0.813348 w_k10_GSK_s1 - w_k10_GSK_s13
                                 + 0.99 b_k10_GSK_s13 - 1.01 s_k10_GSK_s13  = 0
 w_k10_GSK_s14_return_constraint: 0.901904 w_k10_GSK_s1 - w_k10_GSK_s14
                                 + 0.99 b_k10_GSK_s14 - 1.01 s_k10_GSK_s14  = 0
 w_k10_GSK_s15_return_constraint: 1.11676 w_k10_GSK_s1 - w_k10_GSK_s15
                                 + 0.99 b_k10_GSK_s15 - 1.01 s_k10_GSK_s15  = 0
 w_k10_GSK_s16_return_constraint: 0.943849 w_k10_GSK_s1 - w_k10_GSK_s16
                                 + 0.99 b_k10_GSK_s16 - 1.01 s_k10_GSK_s16  = 0
 w_k10_GSK_s17_return_constraint: 0.976813 w_k10_GSK_s1 - w_k10_GSK_s17
                                 + 0.99 b_k10_GSK_s17 - 1.01 s_k10_GSK_s17  = 0
 w_k10_GSK_s18_return_constraint: 0.888735 w_k10_GSK_s1 - w_k10_GSK_s18
                                 + 0.99 b_k10_GSK_s18 - 1.01 s_k10_GSK_s18  = 0
 w_k10_GSK_s19_return_constraint: 0.911249 w_k10_GSK_s1 - w_k10_GSK_s19
                                 + 0.99 b_k10_GSK_s19 - 1.01 s_k10_GSK_s19  = 0
 w_k10_GSK_s20_return_constraint: 0.8918 w_k10_GSK_s2 - w_k10_GSK_s20
                                 + 0.99 b_k10_GSK_s20 - 1.01 s_k10_GSK_s20  = 0
 w_k10_GSK_s21_return_constraint: 1.03371 w_k10_GSK_s2 - w_k10_GSK_s21
                                 + 0.99 b_k10_GSK_s21 - 1.01 s_k10_GSK_s21  = 0
 w_k10_GSK_s22_return_constraint: 0.969777 w_k10_GSK_s2 - w_k10_GSK_s22
                                 + 0.99 b_k10_GSK_s22 - 1.01 s_k10_GSK_s22  = 0
 w_k10_GSK_s23_return_constraint: 1.02552 w_k10_GSK_s2 - w_k10_GSK_s23
                                 + 0.99 b_k10_GSK_s23 - 1.01 s_k10_GSK_s23  = 0
 w_k10_GSK_s24_return_constraint: 1.04184 w_k10_GSK_s2 - w_k10_GSK_s24
                                 + 0.99 b_k10_GSK_s24 - 1.01 s_k10_GSK_s24  = 0
 w_k10_GSK_s25_return_constraint: 0.916796 w_k10_GSK_s2 - w_k10_GSK_s25
                                 + 0.99 b_k10_GSK_s25 - 1.01 s_k10_GSK_s25  = 0
 w_k10_GSK_s26_return_constraint: 0.960173 w_k10_GSK_s2 - w_k10_GSK_s26
                                 + 0.99 b_k10_GSK_s26 - 1.01 s_k10_GSK_s26  = 0
 w_k10_GSK_s27_return_constraint: 0.985045 w_k10_GSK_s2 - w_k10_GSK_s27
                                 + 0.99 b_k10_GSK_s27 - 1.01 s_k10_GSK_s27  = 0
 w_k10_GSK_s28_return_constraint: 1.23193 w_k10_GSK_s2 - w_k10_GSK_s28
                                 + 0.99 b_k10_GSK_s28 - 1.01 s_k10_GSK_s28  = 0
 w_k10_GSK_s29_return_constraint: 1.0122 w_k10_GSK_s2 - w_k10_GSK_s29
                                 + 0.99 b_k10_GSK_s29 - 1.01 s_k10_GSK_s29  = 0
 w_k10_GSK_s30_return_constraint: 1.03658 w_k10_GSK_s3 - w_k10_GSK_s30
                                 + 0.99 b_k10_GSK_s30 - 1.01 s_k10_GSK_s30  = 0
 w_k10_GSK_s31_return_constraint: 0.956207 w_k10_GSK_s3 - w_k10_GSK_s31
                                 + 0.99 b_k10_GSK_s31 - 1.01 s_k10_GSK_s31  = 0
 w_k10_GSK_s32_return_constraint: 1.03453 w_k10_GSK_s3 - w_k10_GSK_s32
                                 + 0.99 b_k10_GSK_s32 - 1.01 s_k10_GSK_s32  = 0
 w_k10_GSK_s33_return_constraint: 0.972242 w_k10_GSK_s3 - w_k10_GSK_s33
                                 + 0.99 b_k10_GSK_s33 - 1.01 s_k10_GSK_s33  = 0
 w_k10_GSK_s34_return_constraint: 1.06515 w_k10_GSK_s3 - w_k10_GSK_s34
                                 + 0.99 b_k10_GSK_s34 - 1.01 s_k10_GSK_s34  = 0
 w_k10_GSK_s35_return_constraint: 1.20982 w_k10_GSK_s3 - w_k10_GSK_s35
                                 + 0.99 b_k10_GSK_s35 - 1.01 s_k10_GSK_s35  = 0
 w_k10_GSK_s36_return_constraint: 0.971408 w_k10_GSK_s3 - w_k10_GSK_s36
                                 + 0.99 b_k10_GSK_s36 - 1.01 s_k10_GSK_s36  = 0
 w_k10_GSK_s37_return_constraint: 0.989684 w_k10_GSK_s3 - w_k10_GSK_s37
                                 + 0.99 b_k10_GSK_s37 - 1.01 s_k10_GSK_s37  = 0
 w_k10_GSK_s38_return_constraint: 1.04232 w_k10_GSK_s3 - w_k10_GSK_s38
                                 + 0.99 b_k10_GSK_s38 - 1.01 s_k10_GSK_s38  = 0
 w_k10_GSK_s39_return_constraint: 0.878451 w_k10_GSK_s3 - w_k10_GSK_s39
                                 + 0.99 b_k10_GSK_s39 - 1.01 s_k10_GSK_s39  = 0
 w_k10_GSK_s40_return_constraint: 0.889576 w_k10_GSK_s4 - w_k10_GSK_s40
                                 + 0.99 b_k10_GSK_s40 - 1.01 s_k10_GSK_s40  = 0
 w_k10_GSK_s41_return_constraint: 0.995661 w_k10_GSK_s4 - w_k10_GSK_s41
                                 + 0.99 b_k10_GSK_s41 - 1.01 s_k10_GSK_s41  = 0
 w_k10_GSK_s42_return_constraint: 1.15811 w_k10_GSK_s4 - w_k10_GSK_s42
                                 + 0.99 b_k10_GSK_s42 - 1.01 s_k10_GSK_s42  = 0
 w_k10_GSK_s43_return_constraint: 0.967866 w_k10_GSK_s4 - w_k10_GSK_s43
                                 + 0.99 b_k10_GSK_s43 - 1.01 s_k10_GSK_s43  = 0
 w_k10_GSK_s44_return_constraint: 1.02986 w_k10_GSK_s4 - w_k10_GSK_s44
                                 + 0.99 b_k10_GSK_s44 - 1.01 s_k10_GSK_s44  = 0
 w_k10_GSK_s45_return_constraint: 0.932107 w_k10_GSK_s4 - w_k10_GSK_s45
                                 + 0.99 b_k10_GSK_s45 - 1.01 s_k10_GSK_s45  = 0
 w_k10_GSK_s46_return_constraint: 1.06697 w_k10_GSK_s4 - w_k10_GSK_s46
                                 + 0.99 b_k10_GSK_s46 - 1.01 s_k10_GSK_s46  = 0
 w_k10_GSK_s47_return_constraint: 1.16059 w_k10_GSK_s4 - w_k10_GSK_s47
                                 + 0.99 b_k10_GSK_s47 - 1.01 s_k10_GSK_s47  = 0
 w_k10_GSK_s48_return_constraint: 1.11815 w_k10_GSK_s4 - w_k10_GSK_s48
                                 + 0.99 b_k10_GSK_s48 - 1.01 s_k10_GSK_s48  = 0
 w_k10_GSK_s49_return_constraint: 0.882615 w_k10_GSK_s4 - w_k10_GSK_s49
                                 + 0.99 b_k10_GSK_s49 - 1.01 s_k10_GSK_s49  = 0
 w_k10_GSK_s50_return_constraint: 1.1646 w_k10_GSK_s5 - w_k10_GSK_s50
                                 + 0.99 b_k10_GSK_s50 - 1.01 s_k10_GSK_s50  = 0
 w_k10_GSK_s51_return_constraint: 0.94886 w_k10_GSK_s5 - w_k10_GSK_s51
                                 + 0.99 b_k10_GSK_s51 - 1.01 s_k10_GSK_s51  = 0
 w_k10_GSK_s52_return_constraint: 0.930643 w_k10_GSK_s5 - w_k10_GSK_s52
                                 + 0.99 b_k10_GSK_s52 - 1.01 s_k10_GSK_s52  = 0
 w_k10_GSK_s53_return_constraint: 0.948189 w_k10_GSK_s5 - w_k10_GSK_s53
                                 + 0.99 b_k10_GSK_s53 - 1.01 s_k10_GSK_s53  = 0
 w_k10_GSK_s54_return_constraint: 0.908219 w_k10_GSK_s5 - w_k10_GSK_s54
                                 + 0.99 b_k10_GSK_s54 - 1.01 s_k10_GSK_s54  = 0
 w_k10_GSK_s55_return_constraint: 0.865603 w_k10_GSK_s5 - w_k10_GSK_s55
                                 + 0.99 b_k10_GSK_s55 - 1.01 s_k10_GSK_s55  = 0
 w_k10_GSK_s56_return_constraint: 1.01836 w_k10_GSK_s5 - w_k10_GSK_s56
                                 + 0.99 b_k10_GSK_s56 - 1.01 s_k10_GSK_s56  = 0
 w_k10_GSK_s57_return_constraint: 0.968341 w_k10_GSK_s5 - w_k10_GSK_s57
                                 + 0.99 b_k10_GSK_s57 - 1.01 s_k10_GSK_s57  = 0
 w_k10_GSK_s58_return_constraint: 1.04068 w_k10_GSK_s5 - w_k10_GSK_s58
                                 + 0.99 b_k10_GSK_s58 - 1.01 s_k10_GSK_s58  = 0
 w_k10_GSK_s59_return_constraint: 1.04238 w_k10_GSK_s5 - w_k10_GSK_s59
                                 + 0.99 b_k10_GSK_s59 - 1.01 s_k10_GSK_s59  = 0
 w_k10_GSK_s60_return_constraint: 0.990482 w_k10_GSK_s6 - w_k10_GSK_s60
                                 + 0.99 b_k10_GSK_s60 - 1.01 s_k10_GSK_s60  = 0
 w_k10_GSK_s61_return_constraint: 1.27037 w_k10_GSK_s6 - w_k10_GSK_s61
                                 + 0.99 b_k10_GSK_s61 - 1.01 s_k10_GSK_s61  = 0
 w_k10_GSK_s62_return_constraint: 1.06955 w_k10_GSK_s6 - w_k10_GSK_s62
                                 + 0.99 b_k10_GSK_s62 - 1.01 s_k10_GSK_s62  = 0
 w_k10_GSK_s63_return_constraint: 0.974796 w_k10_GSK_s6 - w_k10_GSK_s63
                                 + 0.99 b_k10_GSK_s63 - 1.01 s_k10_GSK_s63  = 0
 w_k10_GSK_s64_return_constraint: 0.920248 w_k10_GSK_s6 - w_k10_GSK_s64
                                 + 0.99 b_k10_GSK_s64 - 1.01 s_k10_GSK_s64  = 0
 w_k10_GSK_s65_return_constraint: 0.941162 w_k10_GSK_s6 - w_k10_GSK_s65
                                 + 0.99 b_k10_GSK_s65 - 1.01 s_k10_GSK_s65  = 0
 w_k10_GSK_s66_return_constraint: 1.11823 w_k10_GSK_s6 - w_k10_GSK_s66
                                 + 0.99 b_k10_GSK_s66 - 1.01 s_k10_GSK_s66  = 0
 w_k10_GSK_s67_return_constraint: 1.07185 w_k10_GSK_s6 - w_k10_GSK_s67
                                 + 0.99 b_k10_GSK_s67 - 1.01 s_k10_GSK_s67  = 0
 w_k10_GSK_s68_return_constraint: 1.13955 w_k10_GSK_s6 - w_k10_GSK_s68
                                 + 0.99 b_k10_GSK_s68 - 1.01 s_k10_GSK_s68  = 0
 w_k10_GSK_s69_return_constraint: 1.04244 w_k10_GSK_s6 - w_k10_GSK_s69
                                 + 0.99 b_k10_GSK_s69 - 1.01 s_k10_GSK_s69  = 0
 w_k10_GSK_s70_return_constraint: 0.997366 w_k10_GSK_s7 - w_k10_GSK_s70
                                 + 0.99 b_k10_GSK_s70 - 1.01 s_k10_GSK_s70  = 0
 w_k10_GSK_s71_return_constraint: 1.07278 w_k10_GSK_s7 - w_k10_GSK_s71
                                 + 0.99 b_k10_GSK_s71 - 1.01 s_k10_GSK_s71  = 0
 w_k10_GSK_s72_return_constraint: 0.989946 w_k10_GSK_s7 - w_k10_GSK_s72
                                 + 0.99 b_k10_GSK_s72 - 1.01 s_k10_GSK_s72  = 0
 w_k10_GSK_s73_return_constraint: 1.12033 w_k10_GSK_s7 - w_k10_GSK_s73
                                 + 0.99 b_k10_GSK_s73 - 1.01 s_k10_GSK_s73  = 0
 w_k10_GSK_s74_return_constraint: 0.86708 w_k10_GSK_s7 - w_k10_GSK_s74
                                 + 0.99 b_k10_GSK_s74 - 1.01 s_k10_GSK_s74  = 0
 w_k10_GSK_s75_return_constraint: 1.02714 w_k10_GSK_s7 - w_k10_GSK_s75
                                 + 0.99 b_k10_GSK_s75 - 1.01 s_k10_GSK_s75  = 0
 w_k10_GSK_s76_return_constraint: 0.871067 w_k10_GSK_s7 - w_k10_GSK_s76
                                 + 0.99 b_k10_GSK_s76 - 1.01 s_k10_GSK_s76  = 0
 w_k10_GSK_s77_return_constraint: 1.16973 w_k10_GSK_s7 - w_k10_GSK_s77
                                 + 0.99 b_k10_GSK_s77 - 1.01 s_k10_GSK_s77  = 0
 w_k10_GSK_s78_return_constraint: 0.952751 w_k10_GSK_s7 - w_k10_GSK_s78
                                 + 0.99 b_k10_GSK_s78 - 1.01 s_k10_GSK_s78  = 0
 w_k10_GSK_s79_return_constraint: 1.15807 w_k10_GSK_s7 - w_k10_GSK_s79
                                 + 0.99 b_k10_GSK_s79 - 1.01 s_k10_GSK_s79  = 0
 w_k10_GSK_s80_return_constraint: 1.06378 w_k10_GSK_s8 - w_k10_GSK_s80
                                 + 0.99 b_k10_GSK_s80 - 1.01 s_k10_GSK_s80  = 0
 w_k10_GSK_s81_return_constraint: 0.930239 w_k10_GSK_s8 - w_k10_GSK_s81
                                 + 0.99 b_k10_GSK_s81 - 1.01 s_k10_GSK_s81  = 0
 w_k10_GSK_s82_return_constraint: 1.0961 w_k10_GSK_s8 - w_k10_GSK_s82
                                 + 0.99 b_k10_GSK_s82 - 1.01 s_k10_GSK_s82  = 0
 w_k10_GSK_s83_return_constraint: 1.23859 w_k10_GSK_s8 - w_k10_GSK_s83
                                 + 0.99 b_k10_GSK_s83 - 1.01 s_k10_GSK_s83  = 0
 w_k10_GSK_s84_return_constraint: 0.96204 w_k10_GSK_s8 - w_k10_GSK_s84
                                 + 0.99 b_k10_GSK_s84 - 1.01 s_k10_GSK_s84  = 0
 w_k10_GSK_s85_return_constraint: 0.928281 w_k10_GSK_s8 - w_k10_GSK_s85
                                 + 0.99 b_k10_GSK_s85 - 1.01 s_k10_GSK_s85  = 0
 w_k10_GSK_s86_return_constraint: 0.848395 w_k10_GSK_s8 - w_k10_GSK_s86
                                 + 0.99 b_k10_GSK_s86 - 1.01 s_k10_GSK_s86  = 0
 w_k10_GSK_s87_return_constraint: 1.05643 w_k10_GSK_s8 - w_k10_GSK_s87
                                 + 0.99 b_k10_GSK_s87 - 1.01 s_k10_GSK_s87  = 0
 w_k10_GSK_s88_return_constraint: 0.930265 w_k10_GSK_s8 - w_k10_GSK_s88
                                 + 0.99 b_k10_GSK_s88 - 1.01 s_k10_GSK_s88  = 0
 w_k10_GSK_s89_return_constraint: 0.866849 w_k10_GSK_s8 - w_k10_GSK_s89
                                 + 0.99 b_k10_GSK_s89 - 1.01 s_k10_GSK_s89  = 0
 w_k10_GSK_s90_return_constraint: 1.11314 w_k10_GSK_s9 - w_k10_GSK_s90
                                 + 0.99 b_k10_GSK_s90 - 1.01 s_k10_GSK_s90  = 0
 w_k10_GSK_s91_return_constraint: 0.931724 w_k10_GSK_s9 - w_k10_GSK_s91
                                 + 0.99 b_k10_GSK_s91 - 1.01 s_k10_GSK_s91  = 0
 w_k10_GSK_s92_return_constraint: 1.20991 w_k10_GSK_s9 - w_k10_GSK_s92
                                 + 0.99 b_k10_GSK_s92 - 1.01 s_k10_GSK_s92  = 0
 w_k10_GSK_s93_return_constraint: 1.1048 w_k10_GSK_s9 - w_k10_GSK_s93
                                 + 0.99 b_k10_GSK_s93 - 1.01 s_k10_GSK_s93  = 0
 w_k10_GSK_s94_return_constraint: 0.962499 w_k10_GSK_s9 - w_k10_GSK_s94
                                 + 0.99 b_k10_GSK_s94 - 1.01 s_k10_GSK_s94  = 0
 w_k10_GSK_s95_return_constraint: 0.796716 w_k10_GSK_s9 - w_k10_GSK_s95
                                 + 0.99 b_k10_GSK_s95 - 1.01 s_k10_GSK_s95  = 0
 w_k10_GSK_s96_return_constraint: 0.904893 w_k10_GSK_s9 - w_k10_GSK_s96
                                 + 0.99 b_k10_GSK_s96 - 1.01 s_k10_GSK_s96  = 0
 w_k10_GSK_s97_return_constraint: 0.976578 w_k10_GSK_s9 - w_k10_GSK_s97
                                 + 0.99 b_k10_GSK_s97 - 1.01 s_k10_GSK_s97  = 0
 w_k10_GSK_s98_return_constraint: 0.852577 w_k10_GSK_s9 - w_k10_GSK_s98
                                 + 0.99 b_k10_GSK_s98 - 1.01 s_k10_GSK_s98  = 0
 w_k10_GSK_s99_return_constraint: 1.12302 w_k10_GSK_s9 - w_k10_GSK_s99
                                 + 0.99 b_k10_GSK_s99 - 1.01 s_k10_GSK_s99  = 0
 w_k10_AZN_s0_return_constraint: 1.08319 w_0_AZN - w_k10_AZN_s0
                                 + 0.99 b_k10_AZN_s0 - 1.01 s_k10_AZN_s0  = 0
 w_k10_AZN_s1_return_constraint: 0.981016 w_0_AZN - w_k10_AZN_s1
                                 + 0.99 b_k10_AZN_s1 - 1.01 s_k10_AZN_s1  = 0
 w_k10_AZN_s2_return_constraint: 1.00761 w_0_AZN - w_k10_AZN_s2
                                 + 0.99 b_k10_AZN_s2 - 1.01 s_k10_AZN_s2  = 0
 w_k10_AZN_s3_return_constraint: 1.04217 w_0_AZN - w_k10_AZN_s3
                                 + 0.99 b_k10_AZN_s3 - 1.01 s_k10_AZN_s3  = 0
 w_k10_AZN_s4_return_constraint: 0.987027 w_0_AZN - w_k10_AZN_s4
                                 + 0.99 b_k10_AZN_s4 - 1.01 s_k10_AZN_s4  = 0
 w_k10_AZN_s5_return_constraint: 0.952574 w_0_AZN - w_k10_AZN_s5
                                 + 0.99 b_k10_AZN_s5 - 1.01 s_k10_AZN_s5  = 0
 w_k10_AZN_s6_return_constraint: 0.994866 w_0_AZN - w_k10_AZN_s6
                                 + 0.99 b_k10_AZN_s6 - 1.01 s_k10_AZN_s6  = 0
 w_k10_AZN_s7_return_constraint: 1.0611 w_0_AZN - w_k10_AZN_s7
                                 + 0.99 b_k10_AZN_s7 - 1.01 s_k10_AZN_s7  = 0
 w_k10_AZN_s8_return_constraint: 0.982779 w_0_AZN - w_k10_AZN_s8
                                 + 0.99 b_k10_AZN_s8 - 1.01 s_k10_AZN_s8  = 0
 w_k10_AZN_s9_return_constraint: 0.951262 w_0_AZN - w_k10_AZN_s9
                                 + 0.99 b_k10_AZN_s9 - 1.01 s_k10_AZN_s9  = 0
 w_k10_AZN_s00_return_constraint: 1.02408 w_k10_AZN_s0 - w_k10_AZN_s00
                                 + 0.99 b_k10_AZN_s00 - 1.01 s_k10_AZN_s00  = 0
 w_k10_AZN_s01_return_constraint: 1.02505 w_k10_AZN_s0 - w_k10_AZN_s01
                                 + 0.99 b_k10_AZN_s01 - 1.01 s_k10_AZN_s01  = 0
 w_k10_AZN_s02_return_constraint: 0.903694 w_k10_AZN_s0 - w_k10_AZN_s02
                                 + 0.99 b_k10_AZN_s02 - 1.01 s_k10_AZN_s02  = 0
 w_k10_AZN_s03_return_constraint: 1.09658 w_k10_AZN_s0 - w_k10_AZN_s03
                                 + 0.99 b_k10_AZN_s03 - 1.01 s_k10_AZN_s03  = 0
 w_k10_AZN_s04_return_constraint: 1.0505 w_k10_AZN_s0 - w_k10_AZN_s04
                                 + 0.99 b_k10_AZN_s04 - 1.01 s_k10_AZN_s04  = 0
 w_k10_AZN_s05_return_constraint: 0.861954 w_k10_AZN_s0 - w_k10_AZN_s05
                                 + 0.99 b_k10_AZN_s05 - 1.01 s_k10_AZN_s05  = 0
 w_k10_AZN_s06_return_constraint: 1.04197 w_k10_AZN_s0 - w_k10_AZN_s06
                                 + 0.99 b_k10_AZN_s06 - 1.01 s_k10_AZN_s06  = 0
 w_k10_AZN_s07_return_constraint: 0.901997 w_k10_AZN_s0 - w_k10_AZN_s07
                                 + 0.99 b_k10_AZN_s07 - 1.01 s_k10_AZN_s07  = 0
 w_k10_AZN_s08_return_constraint: 0.935246 w_k10_AZN_s0 - w_k10_AZN_s08
                                 + 0.99 b_k10_AZN_s08 - 1.01 s_k10_AZN_s08  = 0
 w_k10_AZN_s09_return_constraint: 1.00341 w_k10_AZN_s0 - w_k10_AZN_s09
                                 + 0.99 b_k10_AZN_s09 - 1.01 s_k10_AZN_s09  = 0
 w_k10_AZN_s10_return_constraint: 1.00096 w_k10_AZN_s1 - w_k10_AZN_s10
                                 + 0.99 b_k10_AZN_s10 - 1.01 s_k10_AZN_s10  = 0
 w_k10_AZN_s11_return_constraint: 0.961004 w_k10_AZN_s1 - w_k10_AZN_s11
                                 + 0.99 b_k10_AZN_s11 - 1.01 s_k10_AZN_s11  = 0
 w_k10_AZN_s12_return_constraint: 1.02112 w_k10_AZN_s1 - w_k10_AZN_s12
                                 + 0.99 b_k10_AZN_s12 - 1.01 s_k10_AZN_s12  = 0
 w_k10_AZN_s13_return_constraint: 0.869209 w_k10_AZN_s1 - w_k10_AZN_s13
                                 + 0.99 b_k10_AZN_s13 - 1.01 s_k10_AZN_s13  = 0
 w_k10_AZN_s14_return_constraint: 0.97078 w_k10_AZN_s1 - w_k10_AZN_s14
                                 + 0.99 b_k10_AZN_s14 - 1.01 s_k10_AZN_s14  = 0
 w_k10_AZN_s15_return_constraint: 1.02261 w_k10_AZN_s1 - w_k10_AZN_s15
                                 + 0.99 b_k10_AZN_s15 - 1.01 s_k10_AZN_s15  = 0
 w_k10_AZN_s16_return_constraint: 0.99658 w_k10_AZN_s1 - w_k10_AZN_s16
                                 + 0.99 b_k10_AZN_s16 - 1.01 s_k10_AZN_s16  = 0
 w_k10_AZN_s17_return_constraint: 0.966575 w_k10_AZN_s1 - w_k10_AZN_s17
                                 + 0.99 b_k10_AZN_s17 - 1.01 s_k10_AZN_s17  = 0
 w_k10_AZN_s18_return_constraint: 0.85584 w_k10_AZN_s1 - w_k10_AZN_s18
                                 + 0.99 b_k10_AZN_s18 - 1.01 s_k10_AZN_s18  = 0
 w_k10_AZN_s19_return_constraint: 1.04711 w_k10_AZN_s1 - w_k10_AZN_s19
                                 + 0.99 b_k10_AZN_s19 - 1.01 s_k10_AZN_s19  = 0
 w_k10_AZN_s20_return_constraint: 1.00962 w_k10_AZN_s2 - w_k10_AZN_s20
                                 + 0.99 b_k10_AZN_s20 - 1.01 s_k10_AZN_s20  = 0
 w_k10_AZN_s21_return_constraint: 1.04599 w_k10_AZN_s2 - w_k10_AZN_s21
                                 + 0.99 b_k10_AZN_s21 - 1.01 s_k10_AZN_s21  = 0
 w_k10_AZN_s22_return_constraint: 0.94968 w_k10_AZN_s2 - w_k10_AZN_s22
                                 + 0.99 b_k10_AZN_s22 - 1.01 s_k10_AZN_s22  = 0
 w_k10_AZN_s23_return_constraint: 1.08641 w_k10_AZN_s2 - w_k10_AZN_s23
                                 + 0.99 b_k10_AZN_s23 - 1.01 s_k10_AZN_s23  = 0
 w_k10_AZN_s24_return_constraint: 0.964993 w_k10_AZN_s2 - w_k10_AZN_s24
                                 + 0.99 b_k10_AZN_s24 - 1.01 s_k10_AZN_s24  = 0
 w_k10_AZN_s25_return_constraint: 0.963225 w_k10_AZN_s2 - w_k10_AZN_s25
                                 + 0.99 b_k10_AZN_s25 - 1.01 s_k10_AZN_s25  = 0
 w_k10_AZN_s26_return_constraint: 0.947104 w_k10_AZN_s2 - w_k10_AZN_s26
                                 + 0.99 b_k10_AZN_s26 - 1.01 s_k10_AZN_s26  = 0
 w_k10_AZN_s27_return_constraint: 0.954214 w_k10_AZN_s2 - w_k10_AZN_s27
                                 + 0.99 b_k10_AZN_s27 - 1.01 s_k10_AZN_s27  = 0
 w_k10_AZN_s28_return_constraint: 1.06118 w_k10_AZN_s2 - w_k10_AZN_s28
                                 + 0.99 b_k10_AZN_s28 - 1.01 s_k10_AZN_s28  = 0
 w_k10_AZN_s29_return_constraint: 0.997465 w_k10_AZN_s2 - w_k10_AZN_s29
                                 + 0.99 b_k10_AZN_s29 - 1.01 s_k10_AZN_s29  = 0
 w_k10_AZN_s30_return_constraint: 1.11023 w_k10_AZN_s3 - w_k10_AZN_s30
                                 + 0.99 b_k10_AZN_s30 - 1.01 s_k10_AZN_s30  = 0
 w_k10_AZN_s31_return_constraint: 0.979794 w_k10_AZN_s3 - w_k10_AZN_s31
                                 + 0.99 b_k10_AZN_s31 - 1.01 s_k10_AZN_s31  = 0
 w_k10_AZN_s32_return_constraint: 1.0345 w_k10_AZN_s3 - w_k10_AZN_s32
                                 + 0.99 b_k10_AZN_s32 - 1.01 s_k10_AZN_s32  = 0
 w_k10_AZN_s33_return_constraint: 1.02468 w_k10_AZN_s3 - w_k10_AZN_s33
                                 + 0.99 b_k10_AZN_s33 - 1.01 s_k10_AZN_s33  = 0
 w_k10_AZN_s34_return_constraint: 0.970299 w_k10_AZN_s3 - w_k10_AZN_s34
                                 + 0.99 b_k10_AZN_s34 - 1.01 s_k10_AZN_s34  = 0
 w_k10_AZN_s35_return_constraint: 1.12011 w_k10_AZN_s3 - w_k10_AZN_s35
                                 + 0.99 b_k10_AZN_s35 - 1.01 s_k10_AZN_s35  = 0
 w_k10_AZN_s36_return_constraint: 0.866797 w_k10_AZN_s3 - w_k10_AZN_s36
                                 + 0.99 b_k10_AZN_s36 - 1.01 s_k10_AZN_s36  = 0
 w_k10_AZN_s37_return_constraint: 0.999633 w_k10_AZN_s3 - w_k10_AZN_s37
                                 + 0.99 b_k10_AZN_s37 - 1.01 s_k10_AZN_s37  = 0
 w_k10_AZN_s38_return_constraint: 0.987288 w_k10_AZN_s3 - w_k10_AZN_s38
                                 + 0.99 b_k10_AZN_s38 - 1.01 s_k10_AZN_s38  = 0
 w_k10_AZN_s39_return_constraint: 0.984426 w_k10_AZN_s3 - w_k10_AZN_s39
                                 + 0.99 b_k10_AZN_s39 - 1.01 s_k10_AZN_s39  = 0
 w_k10_AZN_s40_return_constraint: 0.980514 w_k10_AZN_s4 - w_k10_AZN_s40
                                 + 0.99 b_k10_AZN_s40 - 1.01 s_k10_AZN_s40  = 0
 w_k10_AZN_s41_return_constraint: 1.00935 w_k10_AZN_s4 - w_k10_AZN_s41
                                 + 0.99 b_k10_AZN_s41 - 1.01 s_k10_AZN_s41  = 0
 w_k10_AZN_s42_return_constraint: 1.09825 w_k10_AZN_s4 - w_k10_AZN_s42
                                 + 0.99 b_k10_AZN_s42 - 1.01 s_k10_AZN_s42  = 0
 w_k10_AZN_s43_return_constraint: 0.964042 w_k10_AZN_s4 - w_k10_AZN_s43
                                 + 0.99 b_k10_AZN_s43 - 1.01 s_k10_AZN_s43  = 0
 w_k10_AZN_s44_return_constraint: 1.02519 w_k10_AZN_s4 - w_k10_AZN_s44
                                 + 0.99 b_k10_AZN_s44 - 1.01 s_k10_AZN_s44  = 0
 w_k10_AZN_s45_return_constraint: 0.987194 w_k10_AZN_s4 - w_k10_AZN_s45
                                 + 0.99 b_k10_AZN_s45 - 1.01 s_k10_AZN_s45  = 0
 w_k10_AZN_s46_return_constraint: 1.04226 w_k10_AZN_s4 - w_k10_AZN_s46
                                 + 0.99 b_k10_AZN_s46 - 1.01 s_k10_AZN_s46  = 0
 w_k10_AZN_s47_return_constraint: 1.10911 w_k10_AZN_s4 - w_k10_AZN_s47
                                 + 0.99 b_k10_AZN_s47 - 1.01 s_k10_AZN_s47  = 0
 w_k10_AZN_s48_return_constraint: 0.999629 w_k10_AZN_s4 - w_k10_AZN_s48
                                 + 0.99 b_k10_AZN_s48 - 1.01 s_k10_AZN_s48  = 0
 w_k10_AZN_s49_return_constraint: 1.04404 w_k10_AZN_s4 - w_k10_AZN_s49
                                 + 0.99 b_k10_AZN_s49 - 1.01 s_k10_AZN_s49  = 0
 w_k10_AZN_s50_return_constraint: 1.06971 w_k10_AZN_s5 - w_k10_AZN_s50
                                 + 0.99 b_k10_AZN_s50 - 1.01 s_k10_AZN_s50  = 0
 w_k10_AZN_s51_return_constraint: 1.00714 w_k10_AZN_s5 - w_k10_AZN_s51
                                 + 0.99 b_k10_AZN_s51 - 1.01 s_k10_AZN_s51  = 0
 w_k10_AZN_s52_return_constraint: 0.990178 w_k10_AZN_s5 - w_k10_AZN_s52
                                 + 0.99 b_k10_AZN_s52 - 1.01 s_k10_AZN_s52  = 0
 w_k10_AZN_s53_return_constraint: 0.986309 w_k10_AZN_s5 - w_k10_AZN_s53
                                 + 0.99 b_k10_AZN_s53 - 1.01 s_k10_AZN_s53  = 0
 w_k10_AZN_s54_return_constraint: 1.00383 w_k10_AZN_s5 - w_k10_AZN_s54
                                 + 0.99 b_k10_AZN_s54 - 1.01 s_k10_AZN_s54  = 0
 w_k10_AZN_s55_return_constraint: 0.937155 w_k10_AZN_s5 - w_k10_AZN_s55
                                 + 0.99 b_k10_AZN_s55 - 1.01 s_k10_AZN_s55  = 0
 w_k10_AZN_s56_return_constraint: 1.06752 w_k10_AZN_s5 - w_k10_AZN_s56
                                 + 0.99 b_k10_AZN_s56 - 1.01 s_k10_AZN_s56  = 0
 w_k10_AZN_s57_return_constraint: 0.913833 w_k10_AZN_s5 - w_k10_AZN_s57
                                 + 0.99 b_k10_AZN_s57 - 1.01 s_k10_AZN_s57  = 0
 w_k10_AZN_s58_return_constraint: 1.00308 w_k10_AZN_s5 - w_k10_AZN_s58
                                 + 0.99 b_k10_AZN_s58 - 1.01 s_k10_AZN_s58  = 0
 w_k10_AZN_s59_return_constraint: 1.07676 w_k10_AZN_s5 - w_k10_AZN_s59
                                 + 0.99 b_k10_AZN_s59 - 1.01 s_k10_AZN_s59  = 0
 w_k10_AZN_s60_return_constraint: 0.925802 w_k10_AZN_s6 - w_k10_AZN_s60
                                 + 0.99 b_k10_AZN_s60 - 1.01 s_k10_AZN_s60  = 0
 w_k10_AZN_s61_return_constraint: 1.00486 w_k10_AZN_s6 - w_k10_AZN_s61
                                 + 0.99 b_k10_AZN_s61 - 1.01 s_k10_AZN_s61  = 0
 w_k10_AZN_s62_return_constraint: 1.05215 w_k10_AZN_s6 - w_k10_AZN_s62
                                 + 0.99 b_k10_AZN_s62 - 1.01 s_k10_AZN_s62  = 0
 w_k10_AZN_s63_return_constraint: 0.94535 w_k10_AZN_s6 - w_k10_AZN_s63
                                 + 0.99 b_k10_AZN_s63 - 1.01 s_k10_AZN_s63  = 0
 w_k10_AZN_s64_return_constraint: 0.917017 w_k10_AZN_s6 - w_k10_AZN_s64
                                 + 0.99 b_k10_AZN_s64 - 1.01 s_k10_AZN_s64  = 0
 w_k10_AZN_s65_return_constraint: 0.934455 w_k10_AZN_s6 - w_k10_AZN_s65
                                 + 0.99 b_k10_AZN_s65 - 1.01 s_k10_AZN_s65  = 0
 w_k10_AZN_s66_return_constraint: 1.02695 w_k10_AZN_s6 - w_k10_AZN_s66
                                 + 0.99 b_k10_AZN_s66 - 1.01 s_k10_AZN_s66  = 0
 w_k10_AZN_s67_return_constraint: 0.945714 w_k10_AZN_s6 - w_k10_AZN_s67
                                 + 0.99 b_k10_AZN_s67 - 1.01 s_k10_AZN_s67  = 0
 w_k10_AZN_s68_return_constraint: 1.01887 w_k10_AZN_s6 - w_k10_AZN_s68
                                 + 0.99 b_k10_AZN_s68 - 1.01 s_k10_AZN_s68  = 0
 w_k10_AZN_s69_return_constraint: 0.893947 w_k10_AZN_s6 - w_k10_AZN_s69
                                 + 0.99 b_k10_AZN_s69 - 1.01 s_k10_AZN_s69  = 0
 w_k10_AZN_s70_return_constraint: 1.02913 w_k10_AZN_s7 - w_k10_AZN_s70
                                 + 0.99 b_k10_AZN_s70 - 1.01 s_k10_AZN_s70  = 0
 w_k10_AZN_s71_return_constraint: 0.98967 w_k10_AZN_s7 - w_k10_AZN_s71
                                 + 0.99 b_k10_AZN_s71 - 1.01 s_k10_AZN_s71  = 0
 w_k10_AZN_s72_return_constraint: 1.06032 w_k10_AZN_s7 - w_k10_AZN_s72
                                 + 0.99 b_k10_AZN_s72 - 1.01 s_k10_AZN_s72  = 0
 w_k10_AZN_s73_return_constraint: 1.10324 w_k10_AZN_s7 - w_k10_AZN_s73
                                 + 0.99 b_k10_AZN_s73 - 1.01 s_k10_AZN_s73  = 0
 w_k10_AZN_s74_return_constraint: 0.920872 w_k10_AZN_s7 - w_k10_AZN_s74
                                 + 0.99 b_k10_AZN_s74 - 1.01 s_k10_AZN_s74  = 0
 w_k10_AZN_s75_return_constraint: 1.06093 w_k10_AZN_s7 - w_k10_AZN_s75
                                 + 0.99 b_k10_AZN_s75 - 1.01 s_k10_AZN_s75  = 0
 w_k10_AZN_s76_return_constraint: 0.952662 w_k10_AZN_s7 - w_k10_AZN_s76
                                 + 0.99 b_k10_AZN_s76 - 1.01 s_k10_AZN_s76  = 0
 w_k10_AZN_s77_return_constraint: 1.1178 w_k10_AZN_s7 - w_k10_AZN_s77
                                 + 0.99 b_k10_AZN_s77 - 1.01 s_k10_AZN_s77  = 0
 w_k10_AZN_s78_return_constraint: 0.952628 w_k10_AZN_s7 - w_k10_AZN_s78
                                 + 0.99 b_k10_AZN_s78 - 1.01 s_k10_AZN_s78  = 0
 w_k10_AZN_s79_return_constraint: 1.00271 w_k10_AZN_s7 - w_k10_AZN_s79
                                 + 0.99 b_k10_AZN_s79 - 1.01 s_k10_AZN_s79  = 0
 w_k10_AZN_s80_return_constraint: 1.02751 w_k10_AZN_s8 - w_k10_AZN_s80
                                 + 0.99 b_k10_AZN_s80 - 1.01 s_k10_AZN_s80  = 0
 w_k10_AZN_s81_return_constraint: 0.998354 w_k10_AZN_s8 - w_k10_AZN_s81
                                 + 0.99 b_k10_AZN_s81 - 1.01 s_k10_AZN_s81  = 0
 w_k10_AZN_s82_return_constraint: 1.04279 w_k10_AZN_s8 - w_k10_AZN_s82
                                 + 0.99 b_k10_AZN_s82 - 1.01 s_k10_AZN_s82  = 0
 w_k10_AZN_s83_return_constraint: 1.0725 w_k10_AZN_s8 - w_k10_AZN_s83
                                 + 0.99 b_k10_AZN_s83 - 1.01 s_k10_AZN_s83  = 0
 w_k10_AZN_s84_return_constraint: 0.978244 w_k10_AZN_s8 - w_k10_AZN_s84
                                 + 0.99 b_k10_AZN_s84 - 1.01 s_k10_AZN_s84  = 0
 w_k10_AZN_s85_return_constraint: 0.944554 w_k10_AZN_s8 - w_k10_AZN_s85
                                 + 0.99 b_k10_AZN_s85 - 1.01 s_k10_AZN_s85  = 0
 w_k10_AZN_s86_return_constraint: 0.932367 w_k10_AZN_s8 - w_k10_AZN_s86
                                 + 0.99 b_k10_AZN_s86 - 1.01 s_k10_AZN_s86  = 0
 w_k10_AZN_s87_return_constraint: 0.97577 w_k10_AZN_s8 - w_k10_AZN_s87
                                 + 0.99 b_k10_AZN_s87 - 1.01 s_k10_AZN_s87  = 0
 w_k10_AZN_s88_return_constraint: 0.99308 w_k10_AZN_s8 - w_k10_AZN_s88
                                 + 0.99 b_k10_AZN_s88 - 1.01 s_k10_AZN_s88  = 0
 w_k10_AZN_s89_return_constraint: 0.864643 w_k10_AZN_s8 - w_k10_AZN_s89
                                 + 0.99 b_k10_AZN_s89 - 1.01 s_k10_AZN_s89  = 0
 w_k10_AZN_s90_return_constraint: 1.01628 w_k10_AZN_s9 - w_k10_AZN_s90
                                 + 0.99 b_k10_AZN_s90 - 1.01 s_k10_AZN_s90  = 0
 w_k10_AZN_s91_return_constraint: 0.993858 w_k10_AZN_s9 - w_k10_AZN_s91
                                 + 0.99 b_k10_AZN_s91 - 1.01 s_k10_AZN_s91  = 0
 w_k10_AZN_s92_return_constraint: 1.07927 w_k10_AZN_s9 - w_k10_AZN_s92
                                 + 0.99 b_k10_AZN_s92 - 1.01 s_k10_AZN_s92  = 0
 w_k10_AZN_s93_return_constraint: 0.986505 w_k10_AZN_s9 - w_k10_AZN_s93
                                 + 0.99 b_k10_AZN_s93 - 1.01 s_k10_AZN_s93  = 0
 w_k10_AZN_s94_return_constraint: 1.10662 w_k10_AZN_s9 - w_k10_AZN_s94
                                 + 0.99 b_k10_AZN_s94 - 1.01 s_k10_AZN_s94  = 0
 w_k10_AZN_s95_return_constraint: 0.944638 w_k10_AZN_s9 - w_k10_AZN_s95
                                 + 0.99 b_k10_AZN_s95 - 1.01 s_k10_AZN_s95  = 0
 w_k10_AZN_s96_return_constraint: 0.948736 w_k10_AZN_s9 - w_k10_AZN_s96
                                 + 0.99 b_k10_AZN_s96 - 1.01 s_k10_AZN_s96  = 0
 w_k10_AZN_s97_return_constraint: 0.957058 w_k10_AZN_s9 - w_k10_AZN_s97
                                 + 0.99 b_k10_AZN_s97 - 1.01 s_k10_AZN_s97  = 0
 w_k10_AZN_s98_return_constraint: 0.931044 w_k10_AZN_s9 - w_k10_AZN_s98
                                 + 0.99 b_k10_AZN_s98 - 1.01 s_k10_AZN_s98  = 0
 w_k10_AZN_s99_return_constraint: 1.05395 w_k10_AZN_s9 - w_k10_AZN_s99
                                 + 0.99 b_k10_AZN_s99 - 1.01 s_k10_AZN_s99  = 0
 w_k10_RIO_s0_return_constraint: 1.17849 w_0_RIO - w_k10_RIO_s0
                                 + 0.99 b_k10_RIO_s0 - 1.01 s_k10_RIO_s0  = 0
 w_k10_RIO_s1_return_constraint: 0.91554 w_0_RIO - w_k10_RIO_s1
                                 + 0.99 b_k10_RIO_s1 - 1.01 s_k10_RIO_s1  = 0
 w_k10_RIO_s2_return_constraint: 1.12655 w_0_RIO - w_k10_RIO_s2
                                 + 0.99 b_k10_RIO_s2 - 1.01 s_k10_RIO_s2  = 0
 w_k10_RIO_s3_return_constraint: 0.938555 w_0_RIO - w_k10_RIO_s3
                                 + 0.99 b_k10_RIO_s3 - 1.01 s_k10_RIO_s3  = 0
 w_k10_RIO_s4_return_constraint: 1.00709 w_0_RIO - w_k10_RIO_s4
                                 + 0.99 b_k10_RIO_s4 - 1.01 s_k10_RIO_s4  = 0
 w_k10_RIO_s5_return_constraint: 0.87773 w_0_RIO - w_k10_RIO_s5
                                 + 0.99 b_k10_RIO_s5 - 1.01 s_k10_RIO_s5  = 0
 w_k10_RIO_s6_return_constraint: 0.839349 w_0_RIO - w_k10_RIO_s6
                                 + 0.99 b_k10_RIO_s6 - 1.01 s_k10_RIO_s6  = 0
 w_k10_RIO_s7_return_constraint: 1.05433 w_0_RIO - w_k10_RIO_s7
                                 + 0.99 b_k10_RIO_s7 - 1.01 s_k10_RIO_s7  = 0
 w_k10_RIO_s8_return_constraint: 0.971742 w_0_RIO - w_k10_RIO_s8
                                 + 0.99 b_k10_RIO_s8 - 1.01 s_k10_RIO_s8  = 0
 w_k10_RIO_s9_return_constraint: 0.852337 w_0_RIO - w_k10_RIO_s9
                                 + 0.99 b_k10_RIO_s9 - 1.01 s_k10_RIO_s9  = 0
 w_k10_RIO_s00_return_constraint: 0.924593 w_k10_RIO_s0 - w_k10_RIO_s00
                                 + 0.99 b_k10_RIO_s00 - 1.01 s_k10_RIO_s00  = 0
 w_k10_RIO_s01_return_constraint: 1.16184 w_k10_RIO_s0 - w_k10_RIO_s01
                                 + 0.99 b_k10_RIO_s01 - 1.01 s_k10_RIO_s01  = 0
 w_k10_RIO_s02_return_constraint: 1.13516 w_k10_RIO_s0 - w_k10_RIO_s02
                                 + 0.99 b_k10_RIO_s02 - 1.01 s_k10_RIO_s02  = 0
 w_k10_RIO_s03_return_constraint: 1.37423 w_k10_RIO_s0 - w_k10_RIO_s03
                                 + 0.99 b_k10_RIO_s03 - 1.01 s_k10_RIO_s03  = 0
 w_k10_RIO_s04_return_constraint: 0.963354 w_k10_RIO_s0 - w_k10_RIO_s04
                                 + 0.99 b_k10_RIO_s04 - 1.01 s_k10_RIO_s04  = 0
 w_k10_RIO_s05_return_constraint: 0.776999 w_k10_RIO_s0 - w_k10_RIO_s05
                                 + 0.99 b_k10_RIO_s05 - 1.01 s_k10_RIO_s05  = 0
 w_k10_RIO_s06_return_constraint: 0.993713 w_k10_RIO_s0 - w_k10_RIO_s06
                                 + 0.99 b_k10_RIO_s06 - 1.01 s_k10_RIO_s06  = 0
 w_k10_RIO_s07_return_constraint: 0.966521 w_k10_RIO_s0 - w_k10_RIO_s07
                                 + 0.99 b_k10_RIO_s07 - 1.01 s_k10_RIO_s07  = 0
 w_k10_RIO_s08_return_constraint: 1.11263 w_k10_RIO_s0 - w_k10_RIO_s08
                                 + 0.99 b_k10_RIO_s08 - 1.01 s_k10_RIO_s08  = 0
 w_k10_RIO_s09_return_constraint: 1.13159 w_k10_RIO_s0 - w_k10_RIO_s09
                                 + 0.99 b_k10_RIO_s09 - 1.01 s_k10_RIO_s09  = 0
 w_k10_RIO_s10_return_constraint: 1.13493 w_k10_RIO_s1 - w_k10_RIO_s10
                                 + 0.99 b_k10_RIO_s10 - 1.01 s_k10_RIO_s10  = 0
 w_k10_RIO_s11_return_constraint: 0.827941 w_k10_RIO_s1 - w_k10_RIO_s11
                                 + 0.99 b_k10_RIO_s11 - 1.01 s_k10_RIO_s11  = 0
 w_k10_RIO_s12_return_constraint: 1.05072 w_k10_RIO_s1 - w_k10_RIO_s12
                                 + 0.99 b_k10_RIO_s12 - 1.01 s_k10_RIO_s12  = 0
 w_k10_RIO_s13_return_constraint: 0.748787 w_k10_RIO_s1 - w_k10_RIO_s13
                                 + 0.99 b_k10_RIO_s13 - 1.01 s_k10_RIO_s13  = 0
 w_k10_RIO_s14_return_constraint: 0.915801 w_k10_RIO_s1 - w_k10_RIO_s14
                                 + 0.99 b_k10_RIO_s14 - 1.01 s_k10_RIO_s14  = 0
 w_k10_RIO_s15_return_constraint: 1.20116 w_k10_RIO_s1 - w_k10_RIO_s15
                                 + 0.99 b_k10_RIO_s15 - 1.01 s_k10_RIO_s15  = 0
 w_k10_RIO_s16_return_constraint: 0.912293 w_k10_RIO_s1 - w_k10_RIO_s16
                                 + 0.99 b_k10_RIO_s16 - 1.01 s_k10_RIO_s16  = 0
 w_k10_RIO_s17_return_constraint: 0.96408 w_k10_RIO_s1 - w_k10_RIO_s17
                                 + 0.99 b_k10_RIO_s17 - 1.01 s_k10_RIO_s17  = 0
 w_k10_RIO_s18_return_constraint: 0.801867 w_k10_RIO_s1 - w_k10_RIO_s18
                                 + 0.99 b_k10_RIO_s18 - 1.01 s_k10_RIO_s18  = 0
 w_k10_RIO_s19_return_constraint: 0.872515 w_k10_RIO_s1 - w_k10_RIO_s19
                                 + 0.99 b_k10_RIO_s19 - 1.01 s_k10_RIO_s19  = 0
 w_k10_RIO_s20_return_constraint: 0.850492 w_k10_RIO_s2 - w_k10_RIO_s20
                                 + 0.99 b_k10_RIO_s20 - 1.01 s_k10_RIO_s20  = 0
 w_k10_RIO_s21_return_constraint: 0.912908 w_k10_RIO_s2 - w_k10_RIO_s21
                                 + 0.99 b_k10_RIO_s21 - 1.01 s_k10_RIO_s21  = 0
 w_k10_RIO_s22_return_constraint: 1.02215 w_k10_RIO_s2 - w_k10_RIO_s22
                                 + 0.99 b_k10_RIO_s22 - 1.01 s_k10_RIO_s22  = 0
 w_k10_RIO_s23_return_constraint: 1.1935 w_k10_RIO_s2 - w_k10_RIO_s23
                                 + 0.99 b_k10_RIO_s23 - 1.01 s_k10_RIO_s23  = 0
 w_k10_RIO_s24_return_constraint: 1.15643 w_k10_RIO_s2 - w_k10_RIO_s24
                                 + 0.99 b_k10_RIO_s24 - 1.01 s_k10_RIO_s24  = 0
 w_k10_RIO_s25_return_constraint: 0.945255 w_k10_RIO_s2 - w_k10_RIO_s25
                                 + 0.99 b_k10_RIO_s25 - 1.01 s_k10_RIO_s25  = 0
 w_k10_RIO_s26_return_constraint: 0.877527 w_k10_RIO_s2 - w_k10_RIO_s26
                                 + 0.99 b_k10_RIO_s26 - 1.01 s_k10_RIO_s26  = 0
 w_k10_RIO_s27_return_constraint: 1.03936 w_k10_RIO_s2 - w_k10_RIO_s27
                                 + 0.99 b_k10_RIO_s27 - 1.01 s_k10_RIO_s27  = 0
 w_k10_RIO_s28_return_constraint: 1.34395 w_k10_RIO_s2 - w_k10_RIO_s28
                                 + 0.99 b_k10_RIO_s28 - 1.01 s_k10_RIO_s28  = 0
 w_k10_RIO_s29_return_constraint: 0.978572 w_k10_RIO_s2 - w_k10_RIO_s29
                                 + 0.99 b_k10_RIO_s29 - 1.01 s_k10_RIO_s29  = 0
 w_k10_RIO_s30_return_constraint: 1.1121 w_k10_RIO_s3 - w_k10_RIO_s30
                                 + 0.99 b_k10_RIO_s30 - 1.01 s_k10_RIO_s30  = 0
 w_k10_RIO_s31_return_constraint: 0.943905 w_k10_RIO_s3 - w_k10_RIO_s31
                                 + 0.99 b_k10_RIO_s31 - 1.01 s_k10_RIO_s31  = 0
 w_k10_RIO_s32_return_constraint: 1.01579 w_k10_RIO_s3 - w_k10_RIO_s32
                                 + 0.99 b_k10_RIO_s32 - 1.01 s_k10_RIO_s32  = 0
 w_k10_RIO_s33_return_constraint: 0.92014 w_k10_RIO_s3 - w_k10_RIO_s33
                                 + 0.99 b_k10_RIO_s33 - 1.01 s_k10_RIO_s33  = 0
 w_k10_RIO_s34_return_constraint: 1.26123 w_k10_RIO_s3 - w_k10_RIO_s34
                                 + 0.99 b_k10_RIO_s34 - 1.01 s_k10_RIO_s34  = 0
 w_k10_RIO_s35_return_constraint: 1.20162 w_k10_RIO_s3 - w_k10_RIO_s35
                                 + 0.99 b_k10_RIO_s35 - 1.01 s_k10_RIO_s35  = 0
 w_k10_RIO_s36_return_constraint: 0.936637 w_k10_RIO_s3 - w_k10_RIO_s36
                                 + 0.99 b_k10_RIO_s36 - 1.01 s_k10_RIO_s36  = 0
 w_k10_RIO_s37_return_constraint: 1.0386 w_k10_RIO_s3 - w_k10_RIO_s37
                                 + 0.99 b_k10_RIO_s37 - 1.01 s_k10_RIO_s37  = 0
 w_k10_RIO_s38_return_constraint: 1.20415 w_k10_RIO_s3 - w_k10_RIO_s38
                                 + 0.99 b_k10_RIO_s38 - 1.01 s_k10_RIO_s38  = 0
 w_k10_RIO_s39_return_constraint: 0.854411 w_k10_RIO_s3 - w_k10_RIO_s39
                                 + 0.99 b_k10_RIO_s39 - 1.01 s_k10_RIO_s39  = 0
 w_k10_RIO_s40_return_constraint: 0.819245 w_k10_RIO_s4 - w_k10_RIO_s40
                                 + 0.99 b_k10_RIO_s40 - 1.01 s_k10_RIO_s40  = 0
 w_k10_RIO_s41_return_constraint: 0.927321 w_k10_RIO_s4 - w_k10_RIO_s41
                                 + 0.99 b_k10_RIO_s41 - 1.01 s_k10_RIO_s41  = 0
 w_k10_RIO_s42_return_constraint: 1.1982 w_k10_RIO_s4 - w_k10_RIO_s42
                                 + 0.99 b_k10_RIO_s42 - 1.01 s_k10_RIO_s42  = 0
 w_k10_RIO_s43_return_constraint: 1.00639 w_k10_RIO_s4 - w_k10_RIO_s43
                                 + 0.99 b_k10_RIO_s43 - 1.01 s_k10_RIO_s43  = 0
 w_k10_RIO_s44_return_constraint: 0.997134 w_k10_RIO_s4 - w_k10_RIO_s44
                                 + 0.99 b_k10_RIO_s44 - 1.01 s_k10_RIO_s44  = 0
 w_k10_RIO_s45_return_constraint: 0.934148 w_k10_RIO_s4 - w_k10_RIO_s45
                                 + 0.99 b_k10_RIO_s45 - 1.01 s_k10_RIO_s45  = 0
 w_k10_RIO_s46_return_constraint: 1.15122 w_k10_RIO_s4 - w_k10_RIO_s46
                                 + 0.99 b_k10_RIO_s46 - 1.01 s_k10_RIO_s46  = 0
 w_k10_RIO_s47_return_constraint: 1.13773 w_k10_RIO_s4 - w_k10_RIO_s47
                                 + 0.99 b_k10_RIO_s47 - 1.01 s_k10_RIO_s47  = 0
 w_k10_RIO_s48_return_constraint: 1.13558 w_k10_RIO_s4 - w_k10_RIO_s48
                                 + 0.99 b_k10_RIO_s48 - 1.01 s_k10_RIO_s48  = 0
 w_k10_RIO_s49_return_constraint: 0.787664 w_k10_RIO_s4 - w_k10_RIO_s49
                                 + 0.99 b_k10_RIO_s49 - 1.01 s_k10_RIO_s49  = 0
 w_k10_RIO_s50_return_constraint: 1.25791 w_k10_RIO_s5 - w_k10_RIO_s50
                                 + 0.99 b_k10_RIO_s50 - 1.01 s_k10_RIO_s50  = 0
 w_k10_RIO_s51_return_constraint: 0.954047 w_k10_RIO_s5 - w_k10_RIO_s51
                                 + 0.99 b_k10_RIO_s51 - 1.01 s_k10_RIO_s51  = 0
 w_k10_RIO_s52_return_constraint: 0.867051 w_k10_RIO_s5 - w_k10_RIO_s52
                                 + 0.99 b_k10_RIO_s52 - 1.01 s_k10_RIO_s52  = 0
 w_k10_RIO_s53_return_constraint: 1.07256 w_k10_RIO_s5 - w_k10_RIO_s53
                                 + 0.99 b_k10_RIO_s53 - 1.01 s_k10_RIO_s53  = 0
 w_k10_RIO_s54_return_constraint: 0.943008 w_k10_RIO_s5 - w_k10_RIO_s54
                                 + 0.99 b_k10_RIO_s54 - 1.01 s_k10_RIO_s54  = 0
 w_k10_RIO_s55_return_constraint: 0.75969 w_k10_RIO_s5 - w_k10_RIO_s55
                                 + 0.99 b_k10_RIO_s55 - 1.01 s_k10_RIO_s55  = 0
 w_k10_RIO_s56_return_constraint: 1.02545 w_k10_RIO_s5 - w_k10_RIO_s56
                                 + 0.99 b_k10_RIO_s56 - 1.01 s_k10_RIO_s56  = 0
 w_k10_RIO_s57_return_constraint: 0.999102 w_k10_RIO_s5 - w_k10_RIO_s57
                                 + 0.99 b_k10_RIO_s57 - 1.01 s_k10_RIO_s57  = 0
 w_k10_RIO_s58_return_constraint: 0.937169 w_k10_RIO_s5 - w_k10_RIO_s58
                                 + 0.99 b_k10_RIO_s58 - 1.01 s_k10_RIO_s58  = 0
 w_k10_RIO_s59_return_constraint: 1.07399 w_k10_RIO_s5 - w_k10_RIO_s59
                                 + 0.99 b_k10_RIO_s59 - 1.01 s_k10_RIO_s59  = 0
 w_k10_RIO_s60_return_constraint: 0.894184 w_k10_RIO_s6 - w_k10_RIO_s60
                                 + 0.99 b_k10_RIO_s60 - 1.01 s_k10_RIO_s60  = 0
 w_k10_RIO_s61_return_constraint: 1.22624 w_k10_RIO_s6 - w_k10_RIO_s61
                                 + 0.99 b_k10_RIO_s61 - 1.01 s_k10_RIO_s61  = 0
 w_k10_RIO_s62_return_constraint: 0.934576 w_k10_RIO_s6 - w_k10_RIO_s62
                                 + 0.99 b_k10_RIO_s62 - 1.01 s_k10_RIO_s62  = 0
 w_k10_RIO_s63_return_constraint: 0.97192 w_k10_RIO_s6 - w_k10_RIO_s63
                                 + 0.99 b_k10_RIO_s63 - 1.01 s_k10_RIO_s63  = 0
 w_k10_RIO_s64_return_constraint: 0.765729 w_k10_RIO_s6 - w_k10_RIO_s64
                                 + 0.99 b_k10_RIO_s64 - 1.01 s_k10_RIO_s64  = 0
 w_k10_RIO_s65_return_constraint: 0.946817 w_k10_RIO_s6 - w_k10_RIO_s65
                                 + 0.99 b_k10_RIO_s65 - 1.01 s_k10_RIO_s65  = 0
 w_k10_RIO_s66_return_constraint: 1.09622 w_k10_RIO_s6 - w_k10_RIO_s66
                                 + 0.99 b_k10_RIO_s66 - 1.01 s_k10_RIO_s66  = 0
 w_k10_RIO_s67_return_constraint: 1.17713 w_k10_RIO_s6 - w_k10_RIO_s67
                                 + 0.99 b_k10_RIO_s67 - 1.01 s_k10_RIO_s67  = 0
 w_k10_RIO_s68_return_constraint: 1.21804 w_k10_RIO_s6 - w_k10_RIO_s68
                                 + 0.99 b_k10_RIO_s68 - 1.01 s_k10_RIO_s68  = 0
 w_k10_RIO_s69_return_constraint: 1.0469 w_k10_RIO_s6 - w_k10_RIO_s69
                                 + 0.99 b_k10_RIO_s69 - 1.01 s_k10_RIO_s69  = 0
 w_k10_RIO_s70_return_constraint: 0.982982 w_k10_RIO_s7 - w_k10_RIO_s70
                                 + 0.99 b_k10_RIO_s70 - 1.01 s_k10_RIO_s70  = 0
 w_k10_RIO_s71_return_constraint: 1.14425 w_k10_RIO_s7 - w_k10_RIO_s71
                                 + 0.99 b_k10_RIO_s71 - 1.01 s_k10_RIO_s71  = 0
 w_k10_RIO_s72_return_constraint: 1.05453 w_k10_RIO_s7 - w_k10_RIO_s72
                                 + 0.99 b_k10_RIO_s72 - 1.01 s_k10_RIO_s72  = 0
 w_k10_RIO_s73_return_constraint: 1.0798 w_k10_RIO_s7 - w_k10_RIO_s73
                                 + 0.99 b_k10_RIO_s73 - 1.01 s_k10_RIO_s73  = 0
 w_k10_RIO_s74_return_constraint: 0.872429 w_k10_RIO_s7 - w_k10_RIO_s74
                                 + 0.99 b_k10_RIO_s74 - 1.01 s_k10_RIO_s74  = 0
 w_k10_RIO_s75_return_constraint: 0.891305 w_k10_RIO_s7 - w_k10_RIO_s75
                                 + 0.99 b_k10_RIO_s75 - 1.01 s_k10_RIO_s75  = 0
 w_k10_RIO_s76_return_constraint: 0.905125 w_k10_RIO_s7 - w_k10_RIO_s76
                                 + 0.99 b_k10_RIO_s76 - 1.01 s_k10_RIO_s76  = 0
 w_k10_RIO_s77_return_constraint: 1.21061 w_k10_RIO_s7 - w_k10_RIO_s77
                                 + 0.99 b_k10_RIO_s77 - 1.01 s_k10_RIO_s77  = 0
 w_k10_RIO_s78_return_constraint: 0.911965 w_k10_RIO_s7 - w_k10_RIO_s78
                                 + 0.99 b_k10_RIO_s78 - 1.01 s_k10_RIO_s78  = 0
 w_k10_RIO_s79_return_constraint: 1.26139 w_k10_RIO_s7 - w_k10_RIO_s79
                                 + 0.99 b_k10_RIO_s79 - 1.01 s_k10_RIO_s79  = 0
 w_k10_RIO_s80_return_constraint: 1.13074 w_k10_RIO_s8 - w_k10_RIO_s80
                                 + 0.99 b_k10_RIO_s80 - 1.01 s_k10_RIO_s80  = 0
 w_k10_RIO_s81_return_constraint: 1.02716 w_k10_RIO_s8 - w_k10_RIO_s81
                                 + 0.99 b_k10_RIO_s81 - 1.01 s_k10_RIO_s81  = 0
 w_k10_RIO_s82_return_constraint: 1.17787 w_k10_RIO_s8 - w_k10_RIO_s82
                                 + 0.99 b_k10_RIO_s82 - 1.01 s_k10_RIO_s82  = 0
 w_k10_RIO_s83_return_constraint: 1.23814 w_k10_RIO_s8 - w_k10_RIO_s83
                                 + 0.99 b_k10_RIO_s83 - 1.01 s_k10_RIO_s83  = 0
 w_k10_RIO_s84_return_constraint: 1.13027 w_k10_RIO_s8 - w_k10_RIO_s84
                                 + 0.99 b_k10_RIO_s84 - 1.01 s_k10_RIO_s84  = 0
 w_k10_RIO_s85_return_constraint: 0.879908 w_k10_RIO_s8 - w_k10_RIO_s85
                                 + 0.99 b_k10_RIO_s85 - 1.01 s_k10_RIO_s85  = 0
 w_k10_RIO_s86_return_constraint: 0.832888 w_k10_RIO_s8 - w_k10_RIO_s86
                                 + 0.99 b_k10_RIO_s86 - 1.01 s_k10_RIO_s86  = 0
 w_k10_RIO_s87_return_constraint: 1.12085 w_k10_RIO_s8 - w_k10_RIO_s87
                                 + 0.99 b_k10_RIO_s87 - 1.01 s_k10_RIO_s87  = 0
 w_k10_RIO_s88_return_constraint: 0.961512 w_k10_RIO_s8 - w_k10_RIO_s88
                                 + 0.99 b_k10_RIO_s88 - 1.01 s_k10_RIO_s88  = 0
 w_k10_RIO_s89_return_constraint: 0.86396 w_k10_RIO_s8 - w_k10_RIO_s89
                                 + 0.99 b_k10_RIO_s89 - 1.01 s_k10_RIO_s89  = 0
 w_k10_RIO_s90_return_constraint: 1.25284 w_k10_RIO_s9 - w_k10_RIO_s90
                                 + 0.99 b_k10_RIO_s90 - 1.01 s_k10_RIO_s90  = 0
 w_k10_RIO_s91_return_constraint: 1.05047 w_k10_RIO_s9 - w_k10_RIO_s91
                                 + 0.99 b_k10_RIO_s91 - 1.01 s_k10_RIO_s91  = 0
 w_k10_RIO_s92_return_constraint: 1.243 w_k10_RIO_s9 - w_k10_RIO_s92
                                 + 0.99 b_k10_RIO_s92 - 1.01 s_k10_RIO_s92  = 0
 w_k10_RIO_s93_return_constraint: 1.19457 w_k10_RIO_s9 - w_k10_RIO_s93
                                 + 0.99 b_k10_RIO_s93 - 1.01 s_k10_RIO_s93  = 0
 w_k10_RIO_s94_return_constraint: 1.00039 w_k10_RIO_s9 - w_k10_RIO_s94
                                 + 0.99 b_k10_RIO_s94 - 1.01 s_k10_RIO_s94  = 0
 w_k10_RIO_s95_return_constraint: 0.846386 w_k10_RIO_s9 - w_k10_RIO_s95
                                 + 0.99 b_k10_RIO_s95 - 1.01 s_k10_RIO_s95  = 0
 w_k10_RIO_s96_return_constraint: 0.942771 w_k10_RIO_s9 - w_k10_RIO_s96
                                 + 0.99 b_k10_RIO_s96 - 1.01 s_k10_RIO_s96  = 0
 w_k10_RIO_s97_return_constraint: 0.999953 w_k10_RIO_s9 - w_k10_RIO_s97
                                 + 0.99 b_k10_RIO_s97 - 1.01 s_k10_RIO_s97  = 0
 w_k10_RIO_s98_return_constraint: 0.942376 w_k10_RIO_s9 - w_k10_RIO_s98
                                 + 0.99 b_k10_RIO_s98 - 1.01 s_k10_RIO_s98  = 0
 w_k10_RIO_s99_return_constraint: 1.1539 w_k10_RIO_s9 - w_k10_RIO_s99
                                 + 0.99 b_k10_RIO_s99 - 1.01 s_k10_RIO_s99  = 0
 w_k10_BG_s0_return_constraint:  1.04264 w_0_BG - w_k10_BG_s0
                                 + 0.99 b_k10_BG_s0 - 1.01 s_k10_BG_s0  = 0
 w_k10_BG_s1_return_constraint:  0.922261 w_0_BG - w_k10_BG_s1
                                 + 0.99 b_k10_BG_s1 - 1.01 s_k10_BG_s1  = 0
 w_k10_BG_s2_return_constraint:  1.1117 w_0_BG - w_k10_BG_s2 + 0.99 b_k10_BG_s2
                                 - 1.01 s_k10_BG_s2  = 0
 w_k10_BG_s3_return_constraint:  0.909821 w_0_BG - w_k10_BG_s3
                                 + 0.99 b_k10_BG_s3 - 1.01 s_k10_BG_s3  = 0
 w_k10_BG_s4_return_constraint:  1.01547 w_0_BG - w_k10_BG_s4
                                 + 0.99 b_k10_BG_s4 - 1.01 s_k10_BG_s4  = 0
 w_k10_BG_s5_return_constraint:  0.99001 w_0_BG - w_k10_BG_s5
                                 + 0.99 b_k10_BG_s5 - 1.01 s_k10_BG_s5  = 0
 w_k10_BG_s6_return_constraint:  0.943261 w_0_BG - w_k10_BG_s6
                                 + 0.99 b_k10_BG_s6 - 1.01 s_k10_BG_s6  = 0
 w_k10_BG_s7_return_constraint:  0.985769 w_0_BG - w_k10_BG_s7
                                 + 0.99 b_k10_BG_s7 - 1.01 s_k10_BG_s7  = 0
 w_k10_BG_s8_return_constraint:  1.0433 w_0_BG - w_k10_BG_s8 + 0.99 b_k10_BG_s8
                                 - 1.01 s_k10_BG_s8  = 0
 w_k10_BG_s9_return_constraint:  0.954513 w_0_BG - w_k10_BG_s9
                                 + 0.99 b_k10_BG_s9 - 1.01 s_k10_BG_s9  = 0
 w_k10_BG_s00_return_constraint: 0.963755 w_k10_BG_s0 - w_k10_BG_s00
                                 + 0.99 b_k10_BG_s00 - 1.01 s_k10_BG_s00  = 0
 w_k10_BG_s01_return_constraint: 1.07188 w_k10_BG_s0 - w_k10_BG_s01
                                 + 0.99 b_k10_BG_s01 - 1.01 s_k10_BG_s01  = 0
 w_k10_BG_s02_return_constraint: 1.1055 w_k10_BG_s0 - w_k10_BG_s02
                                 + 0.99 b_k10_BG_s02 - 1.01 s_k10_BG_s02  = 0
 w_k10_BG_s03_return_constraint: 1.06111 w_k10_BG_s0 - w_k10_BG_s03
                                 + 0.99 b_k10_BG_s03 - 1.01 s_k10_BG_s03  = 0
 w_k10_BG_s04_return_constraint: 1.01285 w_k10_BG_s0 - w_k10_BG_s04
                                 + 0.99 b_k10_BG_s04 - 1.01 s_k10_BG_s04  = 0
 w_k10_BG_s05_return_constraint: 0.918758 w_k10_BG_s0 - w_k10_BG_s05
                                 + 0.99 b_k10_BG_s05 - 1.01 s_k10_BG_s05  = 0
 w_k10_BG_s06_return_constraint: 0.889266 w_k10_BG_s0 - w_k10_BG_s06
                                 + 0.99 b_k10_BG_s06 - 1.01 s_k10_BG_s06  = 0
 w_k10_BG_s07_return_constraint: 0.989654 w_k10_BG_s0 - w_k10_BG_s07
                                 + 0.99 b_k10_BG_s07 - 1.01 s_k10_BG_s07  = 0
 w_k10_BG_s08_return_constraint: 0.977594 w_k10_BG_s0 - w_k10_BG_s08
                                 + 0.99 b_k10_BG_s08 - 1.01 s_k10_BG_s08  = 0
 w_k10_BG_s09_return_constraint: 1.04532 w_k10_BG_s0 - w_k10_BG_s09
                                 + 0.99 b_k10_BG_s09 - 1.01 s_k10_BG_s09  = 0
 w_k10_BG_s10_return_constraint: 1.08129 w_k10_BG_s1 - w_k10_BG_s10
                                 + 0.99 b_k10_BG_s10 - 1.01 s_k10_BG_s10  = 0
 w_k10_BG_s11_return_constraint: 0.873959 w_k10_BG_s1 - w_k10_BG_s11
                                 + 0.99 b_k10_BG_s11 - 1.01 s_k10_BG_s11  = 0
 w_k10_BG_s12_return_constraint: 0.965524 w_k10_BG_s1 - w_k10_BG_s12
                                 + 0.99 b_k10_BG_s12 - 1.01 s_k10_BG_s12  = 0
 w_k10_BG_s13_return_constraint: 0.924295 w_k10_BG_s1 - w_k10_BG_s13
                                 + 0.99 b_k10_BG_s13 - 1.01 s_k10_BG_s13  = 0
 w_k10_BG_s14_return_constraint: 0.997093 w_k10_BG_s1 - w_k10_BG_s14
                                 + 0.99 b_k10_BG_s14 - 1.01 s_k10_BG_s14  = 0
 w_k10_BG_s15_return_constraint: 1.02482 w_k10_BG_s1 - w_k10_BG_s15
                                 + 0.99 b_k10_BG_s15 - 1.01 s_k10_BG_s15  = 0
 w_k10_BG_s16_return_constraint: 1.02491 w_k10_BG_s1 - w_k10_BG_s16
                                 + 0.99 b_k10_BG_s16 - 1.01 s_k10_BG_s16  = 0
 w_k10_BG_s17_return_constraint: 1.15305 w_k10_BG_s1 - w_k10_BG_s17
                                 + 0.99 b_k10_BG_s17 - 1.01 s_k10_BG_s17  = 0
 w_k10_BG_s18_return_constraint: 0.976535 w_k10_BG_s1 - w_k10_BG_s18
                                 + 0.99 b_k10_BG_s18 - 1.01 s_k10_BG_s18  = 0
 w_k10_BG_s19_return_constraint: 0.93747 w_k10_BG_s1 - w_k10_BG_s19
                                 + 0.99 b_k10_BG_s19 - 1.01 s_k10_BG_s19  = 0
 w_k10_BG_s20_return_constraint: 0.918669 w_k10_BG_s2 - w_k10_BG_s20
                                 + 0.99 b_k10_BG_s20 - 1.01 s_k10_BG_s20  = 0
 w_k10_BG_s21_return_constraint: 0.931422 w_k10_BG_s2 - w_k10_BG_s21
                                 + 0.99 b_k10_BG_s21 - 1.01 s_k10_BG_s21  = 0
 w_k10_BG_s22_return_constraint: 1.11173 w_k10_BG_s2 - w_k10_BG_s22
                                 + 0.99 b_k10_BG_s22 - 1.01 s_k10_BG_s22  = 0
 w_k10_BG_s23_return_constraint: 1.03517 w_k10_BG_s2 - w_k10_BG_s23
                                 + 0.99 b_k10_BG_s23 - 1.01 s_k10_BG_s23  = 0
 w_k10_BG_s24_return_constraint: 1.04076 w_k10_BG_s2 - w_k10_BG_s24
                                 + 0.99 b_k10_BG_s24 - 1.01 s_k10_BG_s24  = 0
 w_k10_BG_s25_return_constraint: 0.987133 w_k10_BG_s2 - w_k10_BG_s25
                                 + 0.99 b_k10_BG_s25 - 1.01 s_k10_BG_s25  = 0
 w_k10_BG_s26_return_constraint: 1.00467 w_k10_BG_s2 - w_k10_BG_s26
                                 + 0.99 b_k10_BG_s26 - 1.01 s_k10_BG_s26  = 0
 w_k10_BG_s27_return_constraint: 1.06384 w_k10_BG_s2 - w_k10_BG_s27
                                 + 0.99 b_k10_BG_s27 - 1.01 s_k10_BG_s27  = 0
 w_k10_BG_s28_return_constraint: 1.06003 w_k10_BG_s2 - w_k10_BG_s28
                                 + 0.99 b_k10_BG_s28 - 1.01 s_k10_BG_s28  = 0
 w_k10_BG_s29_return_constraint: 0.977221 w_k10_BG_s2 - w_k10_BG_s29
                                 + 0.99 b_k10_BG_s29 - 1.01 s_k10_BG_s29  = 0
 w_k10_BG_s30_return_constraint: 0.979756 w_k10_BG_s3 - w_k10_BG_s30
                                 + 0.99 b_k10_BG_s30 - 1.01 s_k10_BG_s30  = 0
 w_k10_BG_s31_return_constraint: 0.970953 w_k10_BG_s3 - w_k10_BG_s31
                                 + 0.99 b_k10_BG_s31 - 1.01 s_k10_BG_s31  = 0
 w_k10_BG_s32_return_constraint: 1.088 w_k10_BG_s3 - w_k10_BG_s32
                                 + 0.99 b_k10_BG_s32 - 1.01 s_k10_BG_s32  = 0
 w_k10_BG_s33_return_constraint: 0.936535 w_k10_BG_s3 - w_k10_BG_s33
                                 + 0.99 b_k10_BG_s33 - 1.01 s_k10_BG_s33  = 0
 w_k10_BG_s34_return_constraint: 1.12301 w_k10_BG_s3 - w_k10_BG_s34
                                 + 0.99 b_k10_BG_s34 - 1.01 s_k10_BG_s34  = 0
 w_k10_BG_s35_return_constraint: 1.06435 w_k10_BG_s3 - w_k10_BG_s35
                                 + 0.99 b_k10_BG_s35 - 1.01 s_k10_BG_s35  = 0
 w_k10_BG_s36_return_constraint: 1.08325 w_k10_BG_s3 - w_k10_BG_s36
                                 + 0.99 b_k10_BG_s36 - 1.01 s_k10_BG_s36  = 0
 w_k10_BG_s37_return_constraint: 1.03282 w_k10_BG_s3 - w_k10_BG_s37
                                 + 0.99 b_k10_BG_s37 - 1.01 s_k10_BG_s37  = 0
 w_k10_BG_s38_return_constraint: 1.13743 w_k10_BG_s3 - w_k10_BG_s38
                                 + 0.99 b_k10_BG_s38 - 1.01 s_k10_BG_s38  = 0
 w_k10_BG_s39_return_constraint: 0.983908 w_k10_BG_s3 - w_k10_BG_s39
                                 + 0.99 b_k10_BG_s39 - 1.01 s_k10_BG_s39  = 0
 w_k10_BG_s40_return_constraint: 0.968971 w_k10_BG_s4 - w_k10_BG_s40
                                 + 0.99 b_k10_BG_s40 - 1.01 s_k10_BG_s40  = 0
 w_k10_BG_s41_return_constraint: 0.942116 w_k10_BG_s4 - w_k10_BG_s41
                                 + 0.99 b_k10_BG_s41 - 1.01 s_k10_BG_s41  = 0
 w_k10_BG_s42_return_constraint: 1.08247 w_k10_BG_s4 - w_k10_BG_s42
                                 + 0.99 b_k10_BG_s42 - 1.01 s_k10_BG_s42  = 0
 w_k10_BG_s43_return_constraint: 1.09117 w_k10_BG_s4 - w_k10_BG_s43
                                 + 0.99 b_k10_BG_s43 - 1.01 s_k10_BG_s43  = 0
 w_k10_BG_s44_return_constraint: 0.993336 w_k10_BG_s4 - w_k10_BG_s44
                                 + 0.99 b_k10_BG_s44 - 1.01 s_k10_BG_s44  = 0
 w_k10_BG_s45_return_constraint: 0.877721 w_k10_BG_s4 - w_k10_BG_s45
                                 + 0.99 b_k10_BG_s45 - 1.01 s_k10_BG_s45  = 0
 w_k10_BG_s46_return_constraint: 1.07845 w_k10_BG_s4 - w_k10_BG_s46
                                 + 0.99 b_k10_BG_s46 - 1.01 s_k10_BG_s46  = 0
 w_k10_BG_s47_return_constraint: 0.931415 w_k10_BG_s4 - w_k10_BG_s47
                                 + 0.99 b_k10_BG_s47 - 1.01 s_k10_BG_s47  = 0
 w_k10_BG_s48_return_constraint: 1.0242 w_k10_BG_s4 - w_k10_BG_s48
                                 + 0.99 b_k10_BG_s48 - 1.01 s_k10_BG_s48  = 0
 w_k10_BG_s49_return_constraint: 0.828839 w_k10_BG_s4 - w_k10_BG_s49
                                 + 0.99 b_k10_BG_s49 - 1.01 s_k10_BG_s49  = 0
 w_k10_BG_s50_return_constraint: 1.06662 w_k10_BG_s5 - w_k10_BG_s50
                                 + 0.99 b_k10_BG_s50 - 1.01 s_k10_BG_s50  = 0
 w_k10_BG_s51_return_constraint: 0.927572 w_k10_BG_s5 - w_k10_BG_s51
                                 + 0.99 b_k10_BG_s51 - 1.01 s_k10_BG_s51  = 0
 w_k10_BG_s52_return_constraint: 0.897679 w_k10_BG_s5 - w_k10_BG_s52
                                 + 0.99 b_k10_BG_s52 - 1.01 s_k10_BG_s52  = 0
 w_k10_BG_s53_return_constraint: 1.03574 w_k10_BG_s5 - w_k10_BG_s53
                                 + 0.99 b_k10_BG_s53 - 1.01 s_k10_BG_s53  = 0
 w_k10_BG_s54_return_constraint: 0.897229 w_k10_BG_s5 - w_k10_BG_s54
                                 + 0.99 b_k10_BG_s54 - 1.01 s_k10_BG_s54  = 0
 w_k10_BG_s55_return_constraint: 0.897 w_k10_BG_s5 - w_k10_BG_s55
                                 + 0.99 b_k10_BG_s55 - 1.01 s_k10_BG_s55  = 0
 w_k10_BG_s56_return_constraint: 0.911572 w_k10_BG_s5 - w_k10_BG_s56
                                 + 0.99 b_k10_BG_s56 - 1.01 s_k10_BG_s56  = 0
 w_k10_BG_s57_return_constraint: 1.13651 w_k10_BG_s5 - w_k10_BG_s57
                                 + 0.99 b_k10_BG_s57 - 1.01 s_k10_BG_s57  = 0
 w_k10_BG_s58_return_constraint: 1.074 w_k10_BG_s5 - w_k10_BG_s58
                                 + 0.99 b_k10_BG_s58 - 1.01 s_k10_BG_s58  = 0
 w_k10_BG_s59_return_constraint: 0.995733 w_k10_BG_s5 - w_k10_BG_s59
                                 + 0.99 b_k10_BG_s59 - 1.01 s_k10_BG_s59  = 0
 w_k10_BG_s60_return_constraint: 0.851803 w_k10_BG_s6 - w_k10_BG_s60
                                 + 0.99 b_k10_BG_s60 - 1.01 s_k10_BG_s60  = 0
 w_k10_BG_s61_return_constraint: 1.04438 w_k10_BG_s6 - w_k10_BG_s61
                                 + 0.99 b_k10_BG_s61 - 1.01 s_k10_BG_s61  = 0
 w_k10_BG_s62_return_constraint: 0.879892 w_k10_BG_s6 - w_k10_BG_s62
                                 + 0.99 b_k10_BG_s62 - 1.01 s_k10_BG_s62  = 0
 w_k10_BG_s63_return_constraint: 1.01625 w_k10_BG_s6 - w_k10_BG_s63
                                 + 0.99 b_k10_BG_s63 - 1.01 s_k10_BG_s63  = 0
 w_k10_BG_s64_return_constraint: 0.926419 w_k10_BG_s6 - w_k10_BG_s64
                                 + 0.99 b_k10_BG_s64 - 1.01 s_k10_BG_s64  = 0
 w_k10_BG_s65_return_constraint: 1.00868 w_k10_BG_s6 - w_k10_BG_s65
                                 + 0.99 b_k10_BG_s65 - 1.01 s_k10_BG_s65  = 0
 w_k10_BG_s66_return_constraint: 1.06278 w_k10_BG_s6 - w_k10_BG_s66
                                 + 0.99 b_k10_BG_s66 - 1.01 s_k10_BG_s66  = 0
 w_k10_BG_s67_return_constraint: 1.13641 w_k10_BG_s6 - w_k10_BG_s67
                                 + 0.99 b_k10_BG_s67 - 1.01 s_k10_BG_s67  = 0
 w_k10_BG_s68_return_constraint: 1.04297 w_k10_BG_s6 - w_k10_BG_s68
                                 + 0.99 b_k10_BG_s68 - 1.01 s_k10_BG_s68  = 0
 w_k10_BG_s69_return_constraint: 1.12601 w_k10_BG_s6 - w_k10_BG_s69
                                 + 0.99 b_k10_BG_s69 - 1.01 s_k10_BG_s69  = 0
 w_k10_BG_s70_return_constraint: 0.960501 w_k10_BG_s7 - w_k10_BG_s70
                                 + 0.99 b_k10_BG_s70 - 1.01 s_k10_BG_s70  = 0
 w_k10_BG_s71_return_constraint: 1.0465 w_k10_BG_s7 - w_k10_BG_s71
                                 + 0.99 b_k10_BG_s71 - 1.01 s_k10_BG_s71  = 0
 w_k10_BG_s72_return_constraint: 1.00002 w_k10_BG_s7 - w_k10_BG_s72
                                 + 0.99 b_k10_BG_s72 - 1.01 s_k10_BG_s72  = 0
 w_k10_BG_s73_return_constraint: 0.947371 w_k10_BG_s7 - w_k10_BG_s73
                                 + 0.99 b_k10_BG_s73 - 1.01 s_k10_BG_s73  = 0
 w_k10_BG_s74_return_constraint: 0.982077 w_k10_BG_s7 - w_k10_BG_s74
                                 + 0.99 b_k10_BG_s74 - 1.01 s_k10_BG_s74  = 0
 w_k10_BG_s75_return_constraint: 0.898705 w_k10_BG_s7 - w_k10_BG_s75
                                 + 0.99 b_k10_BG_s75 - 1.01 s_k10_BG_s75  = 0
 w_k10_BG_s76_return_constraint: 0.917136 w_k10_BG_s7 - w_k10_BG_s76
                                 + 0.99 b_k10_BG_s76 - 1.01 s_k10_BG_s76  = 0
 w_k10_BG_s77_return_constraint: 0.98436 w_k10_BG_s7 - w_k10_BG_s77
                                 + 0.99 b_k10_BG_s77 - 1.01 s_k10_BG_s77  = 0
 w_k10_BG_s78_return_constraint: 1.01455 w_k10_BG_s7 - w_k10_BG_s78
                                 + 0.99 b_k10_BG_s78 - 1.01 s_k10_BG_s78  = 0
 w_k10_BG_s79_return_constraint: 1.13193 w_k10_BG_s7 - w_k10_BG_s79
                                 + 0.99 b_k10_BG_s79 - 1.01 s_k10_BG_s79  = 0
 w_k10_BG_s80_return_constraint: 1.02761 w_k10_BG_s8 - w_k10_BG_s80
                                 + 0.99 b_k10_BG_s80 - 1.01 s_k10_BG_s80  = 0
 w_k10_BG_s81_return_constraint: 1.03953 w_k10_BG_s8 - w_k10_BG_s81
                                 + 0.99 b_k10_BG_s81 - 1.01 s_k10_BG_s81  = 0
 w_k10_BG_s82_return_constraint: 1.12946 w_k10_BG_s8 - w_k10_BG_s82
                                 + 0.99 b_k10_BG_s82 - 1.01 s_k10_BG_s82  = 0
 w_k10_BG_s83_return_constraint: 1.03875 w_k10_BG_s8 - w_k10_BG_s83
                                 + 0.99 b_k10_BG_s83 - 1.01 s_k10_BG_s83  = 0
 w_k10_BG_s84_return_constraint: 1.15169 w_k10_BG_s8 - w_k10_BG_s84
                                 + 0.99 b_k10_BG_s84 - 1.01 s_k10_BG_s84  = 0
 w_k10_BG_s85_return_constraint: 0.985147 w_k10_BG_s8 - w_k10_BG_s85
                                 + 0.99 b_k10_BG_s85 - 1.01 s_k10_BG_s85  = 0
 w_k10_BG_s86_return_constraint: 0.890856 w_k10_BG_s8 - w_k10_BG_s86
                                 + 0.99 b_k10_BG_s86 - 1.01 s_k10_BG_s86  = 0
 w_k10_BG_s87_return_constraint: 0.975377 w_k10_BG_s8 - w_k10_BG_s87
                                 + 0.99 b_k10_BG_s87 - 1.01 s_k10_BG_s87  = 0
 w_k10_BG_s88_return_constraint: 0.898609 w_k10_BG_s8 - w_k10_BG_s88
                                 + 0.99 b_k10_BG_s88 - 1.01 s_k10_BG_s88  = 0
 w_k10_BG_s89_return_constraint: 1.0331 w_k10_BG_s8 - w_k10_BG_s89
                                 + 0.99 b_k10_BG_s89 - 1.01 s_k10_BG_s89  = 0
 w_k10_BG_s90_return_constraint: 1.08134 w_k10_BG_s9 - w_k10_BG_s90
                                 + 0.99 b_k10_BG_s90 - 1.01 s_k10_BG_s90  = 0
 w_k10_BG_s91_return_constraint: 0.967497 w_k10_BG_s9 - w_k10_BG_s91
                                 + 0.99 b_k10_BG_s91 - 1.01 s_k10_BG_s91  = 0
 w_k10_BG_s92_return_constraint: 1.09183 w_k10_BG_s9 - w_k10_BG_s92
                                 + 0.99 b_k10_BG_s92 - 1.01 s_k10_BG_s92  = 0
 w_k10_BG_s93_return_constraint: 1.19853 w_k10_BG_s9 - w_k10_BG_s93
                                 + 0.99 b_k10_BG_s93 - 1.01 s_k10_BG_s93  = 0
 w_k10_BG_s94_return_constraint: 0.868663 w_k10_BG_s9 - w_k10_BG_s94
                                 + 0.99 b_k10_BG_s94 - 1.01 s_k10_BG_s94  = 0
 w_k10_BG_s95_return_constraint: 0.914007 w_k10_BG_s9 - w_k10_BG_s95
                                 + 0.99 b_k10_BG_s95 - 1.01 s_k10_BG_s95  = 0
 w_k10_BG_s96_return_constraint: 1.08742 w_k10_BG_s9 - w_k10_BG_s96
                                 + 0.99 b_k10_BG_s96 - 1.01 s_k10_BG_s96  = 0
 w_k10_BG_s97_return_constraint: 1.04154 w_k10_BG_s9 - w_k10_BG_s97
                                 + 0.99 b_k10_BG_s97 - 1.01 s_k10_BG_s97  = 0
 w_k10_BG_s98_return_constraint: 0.962004 w_k10_BG_s9 - w_k10_BG_s98
                                 + 0.99 b_k10_BG_s98 - 1.01 s_k10_BG_s98  = 0
 w_k10_BG_s99_return_constraint: 0.947954 w_k10_BG_s9 - w_k10_BG_s99
                                 + 0.99 b_k10_BG_s99 - 1.01 s_k10_BG_s99  = 0
 w_k10_TSCO_s0_return_constraint: 1.25555 w_0_TSCO - w_k10_TSCO_s0
                                 + 0.99 b_k10_TSCO_s0 - 1.01 s_k10_TSCO_s0  = 0
 w_k10_TSCO_s1_return_constraint: 0.826107 w_0_TSCO - w_k10_TSCO_s1
                                 + 0.99 b_k10_TSCO_s1 - 1.01 s_k10_TSCO_s1  = 0
 w_k10_TSCO_s2_return_constraint: 1.19696 w_0_TSCO - w_k10_TSCO_s2
                                 + 0.99 b_k10_TSCO_s2 - 1.01 s_k10_TSCO_s2  = 0
 w_k10_TSCO_s3_return_constraint: 0.954964 w_0_TSCO - w_k10_TSCO_s3
                                 + 0.99 b_k10_TSCO_s3 - 1.01 s_k10_TSCO_s3  = 0
 w_k10_TSCO_s4_return_constraint: 1.01402 w_0_TSCO - w_k10_TSCO_s4
                                 + 0.99 b_k10_TSCO_s4 - 1.01 s_k10_TSCO_s4  = 0
 w_k10_TSCO_s5_return_constraint: 0.837627 w_0_TSCO - w_k10_TSCO_s5
                                 + 0.99 b_k10_TSCO_s5 - 1.01 s_k10_TSCO_s5  = 0
 w_k10_TSCO_s6_return_constraint: 0.838716 w_0_TSCO - w_k10_TSCO_s6
                                 + 0.99 b_k10_TSCO_s6 - 1.01 s_k10_TSCO_s6  = 0
 w_k10_TSCO_s7_return_constraint: 1.09444 w_0_TSCO - w_k10_TSCO_s7
                                 + 0.99 b_k10_TSCO_s7 - 1.01 s_k10_TSCO_s7  = 0
 w_k10_TSCO_s8_return_constraint: 0.991095 w_0_TSCO - w_k10_TSCO_s8
                                 + 0.99 b_k10_TSCO_s8 - 1.01 s_k10_TSCO_s8  = 0
 w_k10_TSCO_s9_return_constraint: 0.854184 w_0_TSCO - w_k10_TSCO_s9
                                 + 0.99 b_k10_TSCO_s9 - 1.01 s_k10_TSCO_s9  = 0
 w_k10_TSCO_s00_return_constraint: 0.867282 w_k10_TSCO_s0 - w_k10_TSCO_s00
                                 + 0.99 b_k10_TSCO_s00 - 1.01 s_k10_TSCO_s00
                                  = 0
 w_k10_TSCO_s01_return_constraint: 1.19182 w_k10_TSCO_s0 - w_k10_TSCO_s01
                                 + 0.99 b_k10_TSCO_s01 - 1.01 s_k10_TSCO_s01
                                  = 0
 w_k10_TSCO_s02_return_constraint: 1.09719 w_k10_TSCO_s0 - w_k10_TSCO_s02
                                 + 0.99 b_k10_TSCO_s02 - 1.01 s_k10_TSCO_s02
                                  = 0
 w_k10_TSCO_s03_return_constraint: 1.37073 w_k10_TSCO_s0 - w_k10_TSCO_s03
                                 + 0.99 b_k10_TSCO_s03 - 1.01 s_k10_TSCO_s03
                                  = 0
 w_k10_TSCO_s04_return_constraint: 1.08548 w_k10_TSCO_s0 - w_k10_TSCO_s04
                                 + 0.99 b_k10_TSCO_s04 - 1.01 s_k10_TSCO_s04
                                  = 0
 w_k10_TSCO_s05_return_constraint: 0.716298 w_k10_TSCO_s0 - w_k10_TSCO_s05
                                 + 0.99 b_k10_TSCO_s05 - 1.01 s_k10_TSCO_s05
                                  = 0
 w_k10_TSCO_s06_return_constraint: 0.946762 w_k10_TSCO_s0 - w_k10_TSCO_s06
                                 + 0.99 b_k10_TSCO_s06 - 1.01 s_k10_TSCO_s06
                                  = 0
 w_k10_TSCO_s07_return_constraint: 0.949727 w_k10_TSCO_s0 - w_k10_TSCO_s07
                                 + 0.99 b_k10_TSCO_s07 - 1.01 s_k10_TSCO_s07
                                  = 0
 w_k10_TSCO_s08_return_constraint: 1.00568 w_k10_TSCO_s0 - w_k10_TSCO_s08
                                 + 0.99 b_k10_TSCO_s08 - 1.01 s_k10_TSCO_s08
                                  = 0
 w_k10_TSCO_s09_return_constraint: 1.14915 w_k10_TSCO_s0 - w_k10_TSCO_s09
                                 + 0.99 b_k10_TSCO_s09 - 1.01 s_k10_TSCO_s09
                                  = 0
 w_k10_TSCO_s10_return_constraint: 1.13836 w_k10_TSCO_s1 - w_k10_TSCO_s10
                                 + 0.99 b_k10_TSCO_s10 - 1.01 s_k10_TSCO_s10
                                  = 0
 w_k10_TSCO_s11_return_constraint: 0.889623 w_k10_TSCO_s1 - w_k10_TSCO_s11
                                 + 0.99 b_k10_TSCO_s11 - 1.01 s_k10_TSCO_s11
                                  = 0
 w_k10_TSCO_s12_return_constraint: 1.10243 w_k10_TSCO_s1 - w_k10_TSCO_s12
                                 + 0.99 b_k10_TSCO_s12 - 1.01 s_k10_TSCO_s12
                                  = 0
 w_k10_TSCO_s13_return_constraint: 0.703184 w_k10_TSCO_s1 - w_k10_TSCO_s13
                                 + 0.99 b_k10_TSCO_s13 - 1.01 s_k10_TSCO_s13
                                  = 0
 w_k10_TSCO_s14_return_constraint: 0.918527 w_k10_TSCO_s1 - w_k10_TSCO_s14
                                 + 0.99 b_k10_TSCO_s14 - 1.01 s_k10_TSCO_s14
                                  = 0
 w_k10_TSCO_s15_return_constraint: 1.27453 w_k10_TSCO_s1 - w_k10_TSCO_s15
                                 + 0.99 b_k10_TSCO_s15 - 1.01 s_k10_TSCO_s15
                                  = 0
 w_k10_TSCO_s16_return_constraint: 1.01405 w_k10_TSCO_s1 - w_k10_TSCO_s16
                                 + 0.99 b_k10_TSCO_s16 - 1.01 s_k10_TSCO_s16
                                  = 0
 w_k10_TSCO_s17_return_constraint: 1.05766 w_k10_TSCO_s1 - w_k10_TSCO_s17
                                 + 0.99 b_k10_TSCO_s17 - 1.01 s_k10_TSCO_s17
                                  = 0
 w_k10_TSCO_s18_return_constraint: 0.832675 w_k10_TSCO_s1 - w_k10_TSCO_s18
                                 + 0.99 b_k10_TSCO_s18 - 1.01 s_k10_TSCO_s18
                                  = 0
 w_k10_TSCO_s19_return_constraint: 0.873042 w_k10_TSCO_s1 - w_k10_TSCO_s19
                                 + 0.99 b_k10_TSCO_s19 - 1.01 s_k10_TSCO_s19
                                  = 0
 w_k10_TSCO_s20_return_constraint: 0.779837 w_k10_TSCO_s2 - w_k10_TSCO_s20
                                 + 0.99 b_k10_TSCO_s20 - 1.01 s_k10_TSCO_s20
                                  = 0
 w_k10_TSCO_s21_return_constraint: 0.995411 w_k10_TSCO_s2 - w_k10_TSCO_s21
                                 + 0.99 b_k10_TSCO_s21 - 1.01 s_k10_TSCO_s21
                                  = 0
 w_k10_TSCO_s22_return_constraint: 0.94031 w_k10_TSCO_s2 - w_k10_TSCO_s22
                                 + 0.99 b_k10_TSCO_s22 - 1.01 s_k10_TSCO_s22
                                  = 0
 w_k10_TSCO_s23_return_constraint: 1.21378 w_k10_TSCO_s2 - w_k10_TSCO_s23
                                 + 0.99 b_k10_TSCO_s23 - 1.01 s_k10_TSCO_s23
                                  = 0
 w_k10_TSCO_s24_return_constraint: 1.07994 w_k10_TSCO_s2 - w_k10_TSCO_s24
                                 + 0.99 b_k10_TSCO_s24 - 1.01 s_k10_TSCO_s24
                                  = 0
 w_k10_TSCO_s25_return_constraint: 0.869566 w_k10_TSCO_s2 - w_k10_TSCO_s25
                                 + 0.99 b_k10_TSCO_s25 - 1.01 s_k10_TSCO_s25
                                  = 0
 w_k10_TSCO_s26_return_constraint: 0.849825 w_k10_TSCO_s2 - w_k10_TSCO_s26
                                 + 0.99 b_k10_TSCO_s26 - 1.01 s_k10_TSCO_s26
                                  = 0
 w_k10_TSCO_s27_return_constraint: 1.08855 w_k10_TSCO_s2 - w_k10_TSCO_s27
                                 + 0.99 b_k10_TSCO_s27 - 1.01 s_k10_TSCO_s27
                                  = 0
 w_k10_TSCO_s28_return_constraint: 1.31487 w_k10_TSCO_s2 - w_k10_TSCO_s28
                                 + 0.99 b_k10_TSCO_s28 - 1.01 s_k10_TSCO_s28
                                  = 0
 w_k10_TSCO_s29_return_constraint: 0.964945 w_k10_TSCO_s2 - w_k10_TSCO_s29
                                 + 0.99 b_k10_TSCO_s29 - 1.01 s_k10_TSCO_s29
                                  = 0
 w_k10_TSCO_s30_return_constraint: 1.12283 w_k10_TSCO_s3 - w_k10_TSCO_s30
                                 + 0.99 b_k10_TSCO_s30 - 1.01 s_k10_TSCO_s30
                                  = 0
 w_k10_TSCO_s31_return_constraint: 0.91575 w_k10_TSCO_s3 - w_k10_TSCO_s31
                                 + 0.99 b_k10_TSCO_s31 - 1.01 s_k10_TSCO_s31
                                  = 0
 w_k10_TSCO_s32_return_constraint: 1.04959 w_k10_TSCO_s3 - w_k10_TSCO_s32
                                 + 0.99 b_k10_TSCO_s32 - 1.01 s_k10_TSCO_s32
                                  = 0
 w_k10_TSCO_s33_return_constraint: 0.875897 w_k10_TSCO_s3 - w_k10_TSCO_s33
                                 + 0.99 b_k10_TSCO_s33 - 1.01 s_k10_TSCO_s33
                                  = 0
 w_k10_TSCO_s34_return_constraint: 1.25484 w_k10_TSCO_s3 - w_k10_TSCO_s34
                                 + 0.99 b_k10_TSCO_s34 - 1.01 s_k10_TSCO_s34
                                  = 0
 w_k10_TSCO_s35_return_constraint: 1.36003 w_k10_TSCO_s3 - w_k10_TSCO_s35
                                 + 0.99 b_k10_TSCO_s35 - 1.01 s_k10_TSCO_s35
                                  = 0
 w_k10_TSCO_s36_return_constraint: 0.881763 w_k10_TSCO_s3 - w_k10_TSCO_s36
                                 + 0.99 b_k10_TSCO_s36 - 1.01 s_k10_TSCO_s36
                                  = 0
 w_k10_TSCO_s37_return_constraint: 0.961056 w_k10_TSCO_s3 - w_k10_TSCO_s37
                                 + 0.99 b_k10_TSCO_s37 - 1.01 s_k10_TSCO_s37
                                  = 0
 w_k10_TSCO_s38_return_constraint: 1.20801 w_k10_TSCO_s3 - w_k10_TSCO_s38
                                 + 0.99 b_k10_TSCO_s38 - 1.01 s_k10_TSCO_s38
                                  = 0
 w_k10_TSCO_s39_return_constraint: 0.885012 w_k10_TSCO_s3 - w_k10_TSCO_s39
                                 + 0.99 b_k10_TSCO_s39 - 1.01 s_k10_TSCO_s39
                                  = 0
 w_k10_TSCO_s40_return_constraint: 0.828883 w_k10_TSCO_s4 - w_k10_TSCO_s40
                                 + 0.99 b_k10_TSCO_s40 - 1.01 s_k10_TSCO_s40
                                  = 0
 w_k10_TSCO_s41_return_constraint: 0.952301 w_k10_TSCO_s4 - w_k10_TSCO_s41
                                 + 0.99 b_k10_TSCO_s41 - 1.01 s_k10_TSCO_s41
                                  = 0
 w_k10_TSCO_s42_return_constraint: 1.32121 w_k10_TSCO_s4 - w_k10_TSCO_s42
                                 + 0.99 b_k10_TSCO_s42 - 1.01 s_k10_TSCO_s42
                                  = 0
 w_k10_TSCO_s43_return_constraint: 0.900291 w_k10_TSCO_s4 - w_k10_TSCO_s43
                                 + 0.99 b_k10_TSCO_s43 - 1.01 s_k10_TSCO_s43
                                  = 0
 w_k10_TSCO_s44_return_constraint: 0.916522 w_k10_TSCO_s4 - w_k10_TSCO_s44
                                 + 0.99 b_k10_TSCO_s44 - 1.01 s_k10_TSCO_s44
                                  = 0
 w_k10_TSCO_s45_return_constraint: 0.875633 w_k10_TSCO_s4 - w_k10_TSCO_s45
                                 + 0.99 b_k10_TSCO_s45 - 1.01 s_k10_TSCO_s45
                                  = 0
 w_k10_TSCO_s46_return_constraint: 1.16157 w_k10_TSCO_s4 - w_k10_TSCO_s46
                                 + 0.99 b_k10_TSCO_s46 - 1.01 s_k10_TSCO_s46
                                  = 0
 w_k10_TSCO_s47_return_constraint: 1.13817 w_k10_TSCO_s4 - w_k10_TSCO_s47
                                 + 0.99 b_k10_TSCO_s47 - 1.01 s_k10_TSCO_s47
                                  = 0
 w_k10_TSCO_s48_return_constraint: 1.1583 w_k10_TSCO_s4 - w_k10_TSCO_s48
                                 + 0.99 b_k10_TSCO_s48 - 1.01 s_k10_TSCO_s48
                                  = 0
 w_k10_TSCO_s49_return_constraint: 0.779111 w_k10_TSCO_s4 - w_k10_TSCO_s49
                                 + 0.99 b_k10_TSCO_s49 - 1.01 s_k10_TSCO_s49
                                  = 0
 w_k10_TSCO_s50_return_constraint: 1.33291 w_k10_TSCO_s5 - w_k10_TSCO_s50
                                 + 0.99 b_k10_TSCO_s50 - 1.01 s_k10_TSCO_s50
                                  = 0
 w_k10_TSCO_s51_return_constraint: 0.871756 w_k10_TSCO_s5 - w_k10_TSCO_s51
                                 + 0.99 b_k10_TSCO_s51 - 1.01 s_k10_TSCO_s51
                                  = 0
 w_k10_TSCO_s52_return_constraint: 0.806771 w_k10_TSCO_s5 - w_k10_TSCO_s52
                                 + 0.99 b_k10_TSCO_s52 - 1.01 s_k10_TSCO_s52
                                  = 0
 w_k10_TSCO_s53_return_constraint: 1.06918 w_k10_TSCO_s5 - w_k10_TSCO_s53
                                 + 0.99 b_k10_TSCO_s53 - 1.01 s_k10_TSCO_s53
                                  = 0
 w_k10_TSCO_s54_return_constraint: 1.04739 w_k10_TSCO_s5 - w_k10_TSCO_s54
                                 + 0.99 b_k10_TSCO_s54 - 1.01 s_k10_TSCO_s54
                                  = 0
 w_k10_TSCO_s55_return_constraint: 0.725373 w_k10_TSCO_s5 - w_k10_TSCO_s55
                                 + 0.99 b_k10_TSCO_s55 - 1.01 s_k10_TSCO_s55
                                  = 0
 w_k10_TSCO_s56_return_constraint: 1.00852 w_k10_TSCO_s5 - w_k10_TSCO_s56
                                 + 0.99 b_k10_TSCO_s56 - 1.01 s_k10_TSCO_s56
                                  = 0
 w_k10_TSCO_s57_return_constraint: 1.04544 w_k10_TSCO_s5 - w_k10_TSCO_s57
                                 + 0.99 b_k10_TSCO_s57 - 1.01 s_k10_TSCO_s57
                                  = 0
 w_k10_TSCO_s58_return_constraint: 1.04593 w_k10_TSCO_s5 - w_k10_TSCO_s58
                                 + 0.99 b_k10_TSCO_s58 - 1.01 s_k10_TSCO_s58
                                  = 0
 w_k10_TSCO_s59_return_constraint: 1.09086 w_k10_TSCO_s5 - w_k10_TSCO_s59
                                 + 0.99 b_k10_TSCO_s59 - 1.01 s_k10_TSCO_s59
                                  = 0
 w_k10_TSCO_s60_return_constraint: 0.812276 w_k10_TSCO_s6 - w_k10_TSCO_s60
                                 + 0.99 b_k10_TSCO_s60 - 1.01 s_k10_TSCO_s60
                                  = 0
 w_k10_TSCO_s61_return_constraint: 1.24869 w_k10_TSCO_s6 - w_k10_TSCO_s61
                                 + 0.99 b_k10_TSCO_s61 - 1.01 s_k10_TSCO_s61
                                  = 0
 w_k10_TSCO_s62_return_constraint: 0.9726 w_k10_TSCO_s6 - w_k10_TSCO_s62
                                 + 0.99 b_k10_TSCO_s62 - 1.01 s_k10_TSCO_s62
                                  = 0
 w_k10_TSCO_s63_return_constraint: 0.940361 w_k10_TSCO_s6 - w_k10_TSCO_s63
                                 + 0.99 b_k10_TSCO_s63 - 1.01 s_k10_TSCO_s63
                                  = 0
 w_k10_TSCO_s64_return_constraint: 0.755367 w_k10_TSCO_s6 - w_k10_TSCO_s64
                                 + 0.99 b_k10_TSCO_s64 - 1.01 s_k10_TSCO_s64
                                  = 0
 w_k10_TSCO_s65_return_constraint: 0.847572 w_k10_TSCO_s6 - w_k10_TSCO_s65
                                 + 0.99 b_k10_TSCO_s65 - 1.01 s_k10_TSCO_s65
                                  = 0
 w_k10_TSCO_s66_return_constraint: 1.03963 w_k10_TSCO_s6 - w_k10_TSCO_s66
                                 + 0.99 b_k10_TSCO_s66 - 1.01 s_k10_TSCO_s66
                                  = 0
 w_k10_TSCO_s67_return_constraint: 1.02574 w_k10_TSCO_s6 - w_k10_TSCO_s67
                                 + 0.99 b_k10_TSCO_s67 - 1.01 s_k10_TSCO_s67
                                  = 0
 w_k10_TSCO_s68_return_constraint: 1.17963 w_k10_TSCO_s6 - w_k10_TSCO_s68
                                 + 0.99 b_k10_TSCO_s68 - 1.01 s_k10_TSCO_s68
                                  = 0
 w_k10_TSCO_s69_return_constraint: 0.932982 w_k10_TSCO_s6 - w_k10_TSCO_s69
                                 + 0.99 b_k10_TSCO_s69 - 1.01 s_k10_TSCO_s69
                                  = 0
 w_k10_TSCO_s70_return_constraint: 1.02159 w_k10_TSCO_s7 - w_k10_TSCO_s70
                                 + 0.99 b_k10_TSCO_s70 - 1.01 s_k10_TSCO_s70
                                  = 0
 w_k10_TSCO_s71_return_constraint: 1.02493 w_k10_TSCO_s7 - w_k10_TSCO_s71
                                 + 0.99 b_k10_TSCO_s71 - 1.01 s_k10_TSCO_s71
                                  = 0
 w_k10_TSCO_s72_return_constraint: 1.00853 w_k10_TSCO_s7 - w_k10_TSCO_s72
                                 + 0.99 b_k10_TSCO_s72 - 1.01 s_k10_TSCO_s72
                                  = 0
 w_k10_TSCO_s73_return_constraint: 1.11896 w_k10_TSCO_s7 - w_k10_TSCO_s73
                                 + 0.99 b_k10_TSCO_s73 - 1.01 s_k10_TSCO_s73
                                  = 0
 w_k10_TSCO_s74_return_constraint: 0.842729 w_k10_TSCO_s7 - w_k10_TSCO_s74
                                 + 0.99 b_k10_TSCO_s74 - 1.01 s_k10_TSCO_s74
                                  = 0
 w_k10_TSCO_s75_return_constraint: 0.950315 w_k10_TSCO_s7 - w_k10_TSCO_s75
                                 + 0.99 b_k10_TSCO_s75 - 1.01 s_k10_TSCO_s75
                                  = 0
 w_k10_TSCO_s76_return_constraint: 0.809233 w_k10_TSCO_s7 - w_k10_TSCO_s76
                                 + 0.99 b_k10_TSCO_s76 - 1.01 s_k10_TSCO_s76
                                  = 0
 w_k10_TSCO_s77_return_constraint: 1.15699 w_k10_TSCO_s7 - w_k10_TSCO_s77
                                 + 0.99 b_k10_TSCO_s77 - 1.01 s_k10_TSCO_s77
                                  = 0
 w_k10_TSCO_s78_return_constraint: 0.897014 w_k10_TSCO_s7 - w_k10_TSCO_s78
                                 + 0.99 b_k10_TSCO_s78 - 1.01 s_k10_TSCO_s78
                                  = 0
 w_k10_TSCO_s79_return_constraint: 1.22669 w_k10_TSCO_s7 - w_k10_TSCO_s79
                                 + 0.99 b_k10_TSCO_s79 - 1.01 s_k10_TSCO_s79
                                  = 0
 w_k10_TSCO_s80_return_constraint: 1.21307 w_k10_TSCO_s8 - w_k10_TSCO_s80
                                 + 0.99 b_k10_TSCO_s80 - 1.01 s_k10_TSCO_s80
                                  = 0
 w_k10_TSCO_s81_return_constraint: 0.976656 w_k10_TSCO_s8 - w_k10_TSCO_s81
                                 + 0.99 b_k10_TSCO_s81 - 1.01 s_k10_TSCO_s81
                                  = 0
 w_k10_TSCO_s82_return_constraint: 1.26874 w_k10_TSCO_s8 - w_k10_TSCO_s82
                                 + 0.99 b_k10_TSCO_s82 - 1.01 s_k10_TSCO_s82
                                  = 0
 w_k10_TSCO_s83_return_constraint: 1.24619 w_k10_TSCO_s8 - w_k10_TSCO_s83
                                 + 0.99 b_k10_TSCO_s83 - 1.01 s_k10_TSCO_s83
                                  = 0
 w_k10_TSCO_s84_return_constraint: 1.16084 w_k10_TSCO_s8 - w_k10_TSCO_s84
                                 + 0.99 b_k10_TSCO_s84 - 1.01 s_k10_TSCO_s84
                                  = 0
 w_k10_TSCO_s85_return_constraint: 0.908734 w_k10_TSCO_s8 - w_k10_TSCO_s85
                                 + 0.99 b_k10_TSCO_s85 - 1.01 s_k10_TSCO_s85
                                  = 0
 w_k10_TSCO_s86_return_constraint: 0.784718 w_k10_TSCO_s8 - w_k10_TSCO_s86
                                 + 0.99 b_k10_TSCO_s86 - 1.01 s_k10_TSCO_s86
                                  = 0
 w_k10_TSCO_s87_return_constraint: 1.09801 w_k10_TSCO_s8 - w_k10_TSCO_s87
                                 + 0.99 b_k10_TSCO_s87 - 1.01 s_k10_TSCO_s87
                                  = 0
 w_k10_TSCO_s88_return_constraint: 0.878986 w_k10_TSCO_s8 - w_k10_TSCO_s88
                                 + 0.99 b_k10_TSCO_s88 - 1.01 s_k10_TSCO_s88
                                  = 0
 w_k10_TSCO_s89_return_constraint: 0.767235 w_k10_TSCO_s8 - w_k10_TSCO_s89
                                 + 0.99 b_k10_TSCO_s89 - 1.01 s_k10_TSCO_s89
                                  = 0
 w_k10_TSCO_s90_return_constraint: 1.23922 w_k10_TSCO_s9 - w_k10_TSCO_s90
                                 + 0.99 b_k10_TSCO_s90 - 1.01 s_k10_TSCO_s90
                                  = 0
 w_k10_TSCO_s91_return_constraint: 1.08722 w_k10_TSCO_s9 - w_k10_TSCO_s91
                                 + 0.99 b_k10_TSCO_s91 - 1.01 s_k10_TSCO_s91
                                  = 0
 w_k10_TSCO_s92_return_constraint: 1.3837 w_k10_TSCO_s9 - w_k10_TSCO_s92
                                 + 0.99 b_k10_TSCO_s92 - 1.01 s_k10_TSCO_s92
                                  = 0
 w_k10_TSCO_s93_return_constraint: 1.13549 w_k10_TSCO_s9 - w_k10_TSCO_s93
                                 + 0.99 b_k10_TSCO_s93 - 1.01 s_k10_TSCO_s93
                                  = 0
 w_k10_TSCO_s94_return_constraint: 0.912177 w_k10_TSCO_s9 - w_k10_TSCO_s94
                                 + 0.99 b_k10_TSCO_s94 - 1.01 s_k10_TSCO_s94
                                  = 0
 w_k10_TSCO_s95_return_constraint: 0.752705 w_k10_TSCO_s9 - w_k10_TSCO_s95
                                 + 0.99 b_k10_TSCO_s95 - 1.01 s_k10_TSCO_s95
                                  = 0
 w_k10_TSCO_s96_return_constraint: 0.917382 w_k10_TSCO_s9 - w_k10_TSCO_s96
                                 + 0.99 b_k10_TSCO_s96 - 1.01 s_k10_TSCO_s96
                                  = 0
 w_k10_TSCO_s97_return_constraint: 1.00618 w_k10_TSCO_s9 - w_k10_TSCO_s97
                                 + 0.99 b_k10_TSCO_s97 - 1.01 s_k10_TSCO_s97
                                  = 0
 w_k10_TSCO_s98_return_constraint: 0.872372 w_k10_TSCO_s9 - w_k10_TSCO_s98
                                 + 0.99 b_k10_TSCO_s98 - 1.01 s_k10_TSCO_s98
                                  = 0
 w_k10_TSCO_s99_return_constraint: 1.26483 w_k10_TSCO_s9 - w_k10_TSCO_s99
                                 + 0.99 b_k10_TSCO_s99 - 1.01 s_k10_TSCO_s99
                                  = 0
 w_k10_AAL_s0_return_constraint: 1.00621 w_0_AAL - w_k10_AAL_s0
                                 + 0.99 b_k10_AAL_s0 - 1.01 s_k10_AAL_s0  = 0
 w_k10_AAL_s1_return_constraint: 0.953994 w_0_AAL - w_k10_AAL_s1
                                 + 0.99 b_k10_AAL_s1 - 1.01 s_k10_AAL_s1  = 0
 w_k10_AAL_s2_return_constraint: 0.971485 w_0_AAL - w_k10_AAL_s2
                                 + 0.99 b_k10_AAL_s2 - 1.01 s_k10_AAL_s2  = 0
 w_k10_AAL_s3_return_constraint: 1.16526 w_0_AAL - w_k10_AAL_s3
                                 + 0.99 b_k10_AAL_s3 - 1.01 s_k10_AAL_s3  = 0
 w_k10_AAL_s4_return_constraint: 0.876658 w_0_AAL - w_k10_AAL_s4
                                 + 0.99 b_k10_AAL_s4 - 1.01 s_k10_AAL_s4  = 0
 w_k10_AAL_s5_return_constraint: 0.860202 w_0_AAL - w_k10_AAL_s5
                                 + 0.99 b_k10_AAL_s5 - 1.01 s_k10_AAL_s5  = 0
 w_k10_AAL_s6_return_constraint: 1.11 w_0_AAL - w_k10_AAL_s6
                                 + 0.99 b_k10_AAL_s6 - 1.01 s_k10_AAL_s6  = 0
 w_k10_AAL_s7_return_constraint: 1.05609 w_0_AAL - w_k10_AAL_s7
                                 + 0.99 b_k10_AAL_s7 - 1.01 s_k10_AAL_s7  = 0
 w_k10_AAL_s8_return_constraint: 1.01265 w_0_AAL - w_k10_AAL_s8
                                 + 0.99 b_k10_AAL_s8 - 1.01 s_k10_AAL_s8  = 0
 w_k10_AAL_s9_return_constraint: 1.02132 w_0_AAL - w_k10_AAL_s9
                                 + 0.99 b_k10_AAL_s9 - 1.01 s_k10_AAL_s9  = 0
 w_k10_AAL_s00_return_constraint: 0.965946 w_k10_AAL_s0 - w_k10_AAL_s00
                                 + 0.99 b_k10_AAL_s00 - 1.01 s_k10_AAL_s00  = 0
 w_k10_AAL_s01_return_constraint: 0.93101 w_k10_AAL_s0 - w_k10_AAL_s01
                                 + 0.99 b_k10_AAL_s01 - 1.01 s_k10_AAL_s01  = 0
 w_k10_AAL_s02_return_constraint: 0.935703 w_k10_AAL_s0 - w_k10_AAL_s02
                                 + 0.99 b_k10_AAL_s02 - 1.01 s_k10_AAL_s02  = 0
 w_k10_AAL_s03_return_constraint: 1.05764 w_k10_AAL_s0 - w_k10_AAL_s03
                                 + 0.99 b_k10_AAL_s03 - 1.01 s_k10_AAL_s03  = 0
 w_k10_AAL_s04_return_constraint: 1.01975 w_k10_AAL_s0 - w_k10_AAL_s04
                                 + 0.99 b_k10_AAL_s04 - 1.01 s_k10_AAL_s04  = 0
 w_k10_AAL_s05_return_constraint: 0.946041 w_k10_AAL_s0 - w_k10_AAL_s05
                                 + 0.99 b_k10_AAL_s05 - 1.01 s_k10_AAL_s05  = 0
 w_k10_AAL_s06_return_constraint: 1.20716 w_k10_AAL_s0 - w_k10_AAL_s06
                                 + 0.99 b_k10_AAL_s06 - 1.01 s_k10_AAL_s06  = 0
 w_k10_AAL_s07_return_constraint: 0.920512 w_k10_AAL_s0 - w_k10_AAL_s07
                                 + 0.99 b_k10_AAL_s07 - 1.01 s_k10_AAL_s07  = 0
 w_k10_AAL_s08_return_constraint: 0.840397 w_k10_AAL_s0 - w_k10_AAL_s08
                                 + 0.99 b_k10_AAL_s08 - 1.01 s_k10_AAL_s08  = 0
 w_k10_AAL_s09_return_constraint: 1.08435 w_k10_AAL_s0 - w_k10_AAL_s09
                                 + 0.99 b_k10_AAL_s09 - 1.01 s_k10_AAL_s09  = 0
 w_k10_AAL_s10_return_constraint: 0.895276 w_k10_AAL_s1 - w_k10_AAL_s10
                                 + 0.99 b_k10_AAL_s10 - 1.01 s_k10_AAL_s10  = 0
 w_k10_AAL_s11_return_constraint: 1.08242 w_k10_AAL_s1 - w_k10_AAL_s11
                                 + 0.99 b_k10_AAL_s11 - 1.01 s_k10_AAL_s11  = 0
 w_k10_AAL_s12_return_constraint: 1.06545 w_k10_AAL_s1 - w_k10_AAL_s12
                                 + 0.99 b_k10_AAL_s12 - 1.01 s_k10_AAL_s12  = 0
 w_k10_AAL_s13_return_constraint: 1.08034 w_k10_AAL_s1 - w_k10_AAL_s13
                                 + 0.99 b_k10_AAL_s13 - 1.01 s_k10_AAL_s13  = 0
 w_k10_AAL_s14_return_constraint: 0.938595 w_k10_AAL_s1 - w_k10_AAL_s14
                                 + 0.99 b_k10_AAL_s14 - 1.01 s_k10_AAL_s14  = 0
 w_k10_AAL_s15_return_constraint: 0.969449 w_k10_AAL_s1 - w_k10_AAL_s15
                                 + 0.99 b_k10_AAL_s15 - 1.01 s_k10_AAL_s15  = 0
 w_k10_AAL_s16_return_constraint: 1.1617 w_k10_AAL_s1 - w_k10_AAL_s16
                                 + 0.99 b_k10_AAL_s16 - 1.01 s_k10_AAL_s16  = 0
 w_k10_AAL_s17_return_constraint: 0.928179 w_k10_AAL_s1 - w_k10_AAL_s17
                                 + 0.99 b_k10_AAL_s17 - 1.01 s_k10_AAL_s17  = 0
 w_k10_AAL_s18_return_constraint: 0.885122 w_k10_AAL_s1 - w_k10_AAL_s18
                                 + 0.99 b_k10_AAL_s18 - 1.01 s_k10_AAL_s18  = 0
 w_k10_AAL_s19_return_constraint: 1.11697 w_k10_AAL_s1 - w_k10_AAL_s19
                                 + 0.99 b_k10_AAL_s19 - 1.01 s_k10_AAL_s19  = 0
 w_k10_AAL_s20_return_constraint: 0.964082 w_k10_AAL_s2 - w_k10_AAL_s20
                                 + 0.99 b_k10_AAL_s20 - 1.01 s_k10_AAL_s20  = 0
 w_k10_AAL_s21_return_constraint: 1.02029 w_k10_AAL_s2 - w_k10_AAL_s21
                                 + 0.99 b_k10_AAL_s21 - 1.01 s_k10_AAL_s21  = 0
 w_k10_AAL_s22_return_constraint: 0.931614 w_k10_AAL_s2 - w_k10_AAL_s22
                                 + 0.99 b_k10_AAL_s22 - 1.01 s_k10_AAL_s22  = 0
 w_k10_AAL_s23_return_constraint: 1.00576 w_k10_AAL_s2 - w_k10_AAL_s23
                                 + 0.99 b_k10_AAL_s23 - 1.01 s_k10_AAL_s23  = 0
 w_k10_AAL_s24_return_constraint: 0.853138 w_k10_AAL_s2 - w_k10_AAL_s24
                                 + 0.99 b_k10_AAL_s24 - 1.01 s_k10_AAL_s24  = 0
 w_k10_AAL_s25_return_constraint: 1.09433 w_k10_AAL_s2 - w_k10_AAL_s25
                                 + 0.99 b_k10_AAL_s25 - 1.01 s_k10_AAL_s25  = 0
 w_k10_AAL_s26_return_constraint: 0.885311 w_k10_AAL_s2 - w_k10_AAL_s26
                                 + 0.99 b_k10_AAL_s26 - 1.01 s_k10_AAL_s26  = 0
 w_k10_AAL_s27_return_constraint: 0.949654 w_k10_AAL_s2 - w_k10_AAL_s27
                                 + 0.99 b_k10_AAL_s27 - 1.01 s_k10_AAL_s27  = 0
 w_k10_AAL_s28_return_constraint: 1.11516 w_k10_AAL_s2 - w_k10_AAL_s28
                                 + 0.99 b_k10_AAL_s28 - 1.01 s_k10_AAL_s28  = 0
 w_k10_AAL_s29_return_constraint: 1.26373 w_k10_AAL_s2 - w_k10_AAL_s29
                                 + 0.99 b_k10_AAL_s29 - 1.01 s_k10_AAL_s29  = 0
 w_k10_AAL_s30_return_constraint: 1.1485 w_k10_AAL_s3 - w_k10_AAL_s30
                                 + 0.99 b_k10_AAL_s30 - 1.01 s_k10_AAL_s30  = 0
 w_k10_AAL_s31_return_constraint: 0.841039 w_k10_AAL_s3 - w_k10_AAL_s31
                                 + 0.99 b_k10_AAL_s31 - 1.01 s_k10_AAL_s31  = 0
 w_k10_AAL_s32_return_constraint: 1.06573 w_k10_AAL_s3 - w_k10_AAL_s32
                                 + 0.99 b_k10_AAL_s32 - 1.01 s_k10_AAL_s32  = 0
 w_k10_AAL_s33_return_constraint: 1.03489 w_k10_AAL_s3 - w_k10_AAL_s33
                                 + 0.99 b_k10_AAL_s33 - 1.01 s_k10_AAL_s33  = 0
 w_k10_AAL_s34_return_constraint: 0.95889 w_k10_AAL_s3 - w_k10_AAL_s34
                                 + 0.99 b_k10_AAL_s34 - 1.01 s_k10_AAL_s34  = 0
 w_k10_AAL_s35_return_constraint: 0.945138 w_k10_AAL_s3 - w_k10_AAL_s35
                                 + 0.99 b_k10_AAL_s35 - 1.01 s_k10_AAL_s35  = 0
 w_k10_AAL_s36_return_constraint: 0.94757 w_k10_AAL_s3 - w_k10_AAL_s36
                                 + 0.99 b_k10_AAL_s36 - 1.01 s_k10_AAL_s36  = 0
 w_k10_AAL_s37_return_constraint: 0.899949 w_k10_AAL_s3 - w_k10_AAL_s37
                                 + 0.99 b_k10_AAL_s37 - 1.01 s_k10_AAL_s37  = 0
 w_k10_AAL_s38_return_constraint: 0.902043 w_k10_AAL_s3 - w_k10_AAL_s38
                                 + 0.99 b_k10_AAL_s38 - 1.01 s_k10_AAL_s38  = 0
 w_k10_AAL_s39_return_constraint: 0.913669 w_k10_AAL_s3 - w_k10_AAL_s39
                                 + 0.99 b_k10_AAL_s39 - 1.01 s_k10_AAL_s39  = 0
 w_k10_AAL_s40_return_constraint: 0.942732 w_k10_AAL_s4 - w_k10_AAL_s40
                                 + 0.99 b_k10_AAL_s40 - 1.01 s_k10_AAL_s40  = 0
 w_k10_AAL_s41_return_constraint: 1.10895 w_k10_AAL_s4 - w_k10_AAL_s41
                                 + 0.99 b_k10_AAL_s41 - 1.01 s_k10_AAL_s41  = 0
 w_k10_AAL_s42_return_constraint: 0.955475 w_k10_AAL_s4 - w_k10_AAL_s42
                                 + 0.99 b_k10_AAL_s42 - 1.01 s_k10_AAL_s42  = 0
 w_k10_AAL_s43_return_constraint: 0.926828 w_k10_AAL_s4 - w_k10_AAL_s43
                                 + 0.99 b_k10_AAL_s43 - 1.01 s_k10_AAL_s43  = 0
 w_k10_AAL_s44_return_constraint: 1.12337 w_k10_AAL_s4 - w_k10_AAL_s44
                                 + 0.99 b_k10_AAL_s44 - 1.01 s_k10_AAL_s44  = 0
 w_k10_AAL_s45_return_constraint: 1.04142 w_k10_AAL_s4 - w_k10_AAL_s45
                                 + 0.99 b_k10_AAL_s45 - 1.01 s_k10_AAL_s45  = 0
 w_k10_AAL_s46_return_constraint: 0.925969 w_k10_AAL_s4 - w_k10_AAL_s46
                                 + 0.99 b_k10_AAL_s46 - 1.01 s_k10_AAL_s46  = 0
 w_k10_AAL_s47_return_constraint: 1.1503 w_k10_AAL_s4 - w_k10_AAL_s47
                                 + 0.99 b_k10_AAL_s47 - 1.01 s_k10_AAL_s47  = 0
 w_k10_AAL_s48_return_constraint: 1.01969 w_k10_AAL_s4 - w_k10_AAL_s48
                                 + 0.99 b_k10_AAL_s48 - 1.01 s_k10_AAL_s48  = 0
 w_k10_AAL_s49_return_constraint: 1.07693 w_k10_AAL_s4 - w_k10_AAL_s49
                                 + 0.99 b_k10_AAL_s49 - 1.01 s_k10_AAL_s49  = 0
 w_k10_AAL_s50_return_constraint: 0.965432 w_k10_AAL_s5 - w_k10_AAL_s50
                                 + 0.99 b_k10_AAL_s50 - 1.01 s_k10_AAL_s50  = 0
 w_k10_AAL_s51_return_constraint: 0.964507 w_k10_AAL_s5 - w_k10_AAL_s51
                                 + 0.99 b_k10_AAL_s51 - 1.01 s_k10_AAL_s51  = 0
 w_k10_AAL_s52_return_constraint: 1.1719 w_k10_AAL_s5 - w_k10_AAL_s52
                                 + 0.99 b_k10_AAL_s52 - 1.01 s_k10_AAL_s52  = 0
 w_k10_AAL_s53_return_constraint: 0.923061 w_k10_AAL_s5 - w_k10_AAL_s53
                                 + 0.99 b_k10_AAL_s53 - 1.01 s_k10_AAL_s53  = 0
 w_k10_AAL_s54_return_constraint: 1.0711 w_k10_AAL_s5 - w_k10_AAL_s54
                                 + 0.99 b_k10_AAL_s54 - 1.01 s_k10_AAL_s54  = 0
 w_k10_AAL_s55_return_constraint: 1.05095 w_k10_AAL_s5 - w_k10_AAL_s55
                                 + 0.99 b_k10_AAL_s55 - 1.01 s_k10_AAL_s55  = 0
 w_k10_AAL_s56_return_constraint: 1.09381 w_k10_AAL_s5 - w_k10_AAL_s56
                                 + 0.99 b_k10_AAL_s56 - 1.01 s_k10_AAL_s56  = 0
 w_k10_AAL_s57_return_constraint: 0.979548 w_k10_AAL_s5 - w_k10_AAL_s57
                                 + 0.99 b_k10_AAL_s57 - 1.01 s_k10_AAL_s57  = 0
 w_k10_AAL_s58_return_constraint: 0.928178 w_k10_AAL_s5 - w_k10_AAL_s58
                                 + 0.99 b_k10_AAL_s58 - 1.01 s_k10_AAL_s58  = 0
 w_k10_AAL_s59_return_constraint: 1.03028 w_k10_AAL_s5 - w_k10_AAL_s59
                                 + 0.99 b_k10_AAL_s59 - 1.01 s_k10_AAL_s59  = 0
 w_k10_AAL_s60_return_constraint: 0.964973 w_k10_AAL_s6 - w_k10_AAL_s60
                                 + 0.99 b_k10_AAL_s60 - 1.01 s_k10_AAL_s60  = 0
 w_k10_AAL_s61_return_constraint: 1.12877 w_k10_AAL_s6 - w_k10_AAL_s61
                                 + 0.99 b_k10_AAL_s61 - 1.01 s_k10_AAL_s61  = 0
 w_k10_AAL_s62_return_constraint: 1.14531 w_k10_AAL_s6 - w_k10_AAL_s62
                                 + 0.99 b_k10_AAL_s62 - 1.01 s_k10_AAL_s62  = 0
 w_k10_AAL_s63_return_constraint: 1.10035 w_k10_AAL_s6 - w_k10_AAL_s63
                                 + 0.99 b_k10_AAL_s63 - 1.01 s_k10_AAL_s63  = 0
 w_k10_AAL_s64_return_constraint: 1.13841 w_k10_AAL_s6 - w_k10_AAL_s64
                                 + 0.99 b_k10_AAL_s64 - 1.01 s_k10_AAL_s64  = 0
 w_k10_AAL_s65_return_constraint: 0.924666 w_k10_AAL_s6 - w_k10_AAL_s65
                                 + 0.99 b_k10_AAL_s65 - 1.01 s_k10_AAL_s65  = 0
 w_k10_AAL_s66_return_constraint: 0.975406 w_k10_AAL_s6 - w_k10_AAL_s66
                                 + 0.99 b_k10_AAL_s66 - 1.01 s_k10_AAL_s66  = 0
 w_k10_AAL_s67_return_constraint: 0.93821 w_k10_AAL_s6 - w_k10_AAL_s67
                                 + 0.99 b_k10_AAL_s67 - 1.01 s_k10_AAL_s67  = 0
 w_k10_AAL_s68_return_constraint: 0.930938 w_k10_AAL_s6 - w_k10_AAL_s68
                                 + 0.99 b_k10_AAL_s68 - 1.01 s_k10_AAL_s68  = 0
 w_k10_AAL_s69_return_constraint: 0.883996 w_k10_AAL_s6 - w_k10_AAL_s69
                                 + 0.99 b_k10_AAL_s69 - 1.01 s_k10_AAL_s69  = 0
 w_k10_AAL_s70_return_constraint: 1.00638 w_k10_AAL_s7 - w_k10_AAL_s70
                                 + 0.99 b_k10_AAL_s70 - 1.01 s_k10_AAL_s70  = 0
 w_k10_AAL_s71_return_constraint: 1.11592 w_k10_AAL_s7 - w_k10_AAL_s71
                                 + 0.99 b_k10_AAL_s71 - 1.01 s_k10_AAL_s71  = 0
 w_k10_AAL_s72_return_constraint: 1.06069 w_k10_AAL_s7 - w_k10_AAL_s72
                                 + 0.99 b_k10_AAL_s72 - 1.01 s_k10_AAL_s72  = 0
 w_k10_AAL_s73_return_constraint: 1.22262 w_k10_AAL_s7 - w_k10_AAL_s73
                                 + 0.99 b_k10_AAL_s73 - 1.01 s_k10_AAL_s73  = 0
 w_k10_AAL_s74_return_constraint: 1.07755 w_k10_AAL_s7 - w_k10_AAL_s74
                                 + 0.99 b_k10_AAL_s74 - 1.01 s_k10_AAL_s74  = 0
 w_k10_AAL_s75_return_constraint: 1.18798 w_k10_AAL_s7 - w_k10_AAL_s75
                                 + 0.99 b_k10_AAL_s75 - 1.01 s_k10_AAL_s75  = 0
 w_k10_AAL_s76_return_constraint: 1.05132 w_k10_AAL_s7 - w_k10_AAL_s76
                                 + 0.99 b_k10_AAL_s76 - 1.01 s_k10_AAL_s76  = 0
 w_k10_AAL_s77_return_constraint: 0.995821 w_k10_AAL_s7 - w_k10_AAL_s77
                                 + 0.99 b_k10_AAL_s77 - 1.01 s_k10_AAL_s77  = 0
 w_k10_AAL_s78_return_constraint: 0.836733 w_k10_AAL_s7 - w_k10_AAL_s78
                                 + 0.99 b_k10_AAL_s78 - 1.01 s_k10_AAL_s78  = 0
 w_k10_AAL_s79_return_constraint: 0.942105 w_k10_AAL_s7 - w_k10_AAL_s79
                                 + 0.99 b_k10_AAL_s79 - 1.01 s_k10_AAL_s79  = 0
 w_k10_AAL_s80_return_constraint: 1.01394 w_k10_AAL_s8 - w_k10_AAL_s80
                                 + 0.99 b_k10_AAL_s80 - 1.01 s_k10_AAL_s80  = 0
 w_k10_AAL_s81_return_constraint: 0.95232 w_k10_AAL_s8 - w_k10_AAL_s81
                                 + 0.99 b_k10_AAL_s81 - 1.01 s_k10_AAL_s81  = 0
 w_k10_AAL_s82_return_constraint: 0.869979 w_k10_AAL_s8 - w_k10_AAL_s82
                                 + 0.99 b_k10_AAL_s82 - 1.01 s_k10_AAL_s82  = 0
 w_k10_AAL_s83_return_constraint: 1.08606 w_k10_AAL_s8 - w_k10_AAL_s83
                                 + 0.99 b_k10_AAL_s83 - 1.01 s_k10_AAL_s83  = 0
 w_k10_AAL_s84_return_constraint: 0.93762 w_k10_AAL_s8 - w_k10_AAL_s84
                                 + 0.99 b_k10_AAL_s84 - 1.01 s_k10_AAL_s84  = 0
 w_k10_AAL_s85_return_constraint: 1.02013 w_k10_AAL_s8 - w_k10_AAL_s85
                                 + 0.99 b_k10_AAL_s85 - 1.01 s_k10_AAL_s85  = 0
 w_k10_AAL_s86_return_constraint: 1.10277 w_k10_AAL_s8 - w_k10_AAL_s86
                                 + 0.99 b_k10_AAL_s86 - 1.01 s_k10_AAL_s86  = 0
 w_k10_AAL_s87_return_constraint: 0.977863 w_k10_AAL_s8 - w_k10_AAL_s87
                                 + 0.99 b_k10_AAL_s87 - 1.01 s_k10_AAL_s87  = 0
 w_k10_AAL_s88_return_constraint: 1.0222 w_k10_AAL_s8 - w_k10_AAL_s88
                                 + 0.99 b_k10_AAL_s88 - 1.01 s_k10_AAL_s88  = 0
 w_k10_AAL_s89_return_constraint: 0.81759 w_k10_AAL_s8 - w_k10_AAL_s89
                                 + 0.99 b_k10_AAL_s89 - 1.01 s_k10_AAL_s89  = 0
 w_k10_AAL_s90_return_constraint: 1.01582 w_k10_AAL_s9 - w_k10_AAL_s90
                                 + 0.99 b_k10_AAL_s90 - 1.01 s_k10_AAL_s90  = 0
 w_k10_AAL_s91_return_constraint: 0.942239 w_k10_AAL_s9 - w_k10_AAL_s91
                                 + 0.99 b_k10_AAL_s91 - 1.01 s_k10_AAL_s91  = 0
 w_k10_AAL_s92_return_constraint: 0.930684 w_k10_AAL_s9 - w_k10_AAL_s92
                                 + 0.99 b_k10_AAL_s92 - 1.01 s_k10_AAL_s92  = 0
 w_k10_AAL_s93_return_constraint: 0.871388 w_k10_AAL_s9 - w_k10_AAL_s93
                                 + 0.99 b_k10_AAL_s93 - 1.01 s_k10_AAL_s93  = 0
 w_k10_AAL_s94_return_constraint: 1.13212 w_k10_AAL_s9 - w_k10_AAL_s94
                                 + 0.99 b_k10_AAL_s94 - 1.01 s_k10_AAL_s94  = 0
 w_k10_AAL_s95_return_constraint: 1.01258 w_k10_AAL_s9 - w_k10_AAL_s95
                                 + 0.99 b_k10_AAL_s95 - 1.01 s_k10_AAL_s95  = 0
 w_k10_AAL_s96_return_constraint: 0.900428 w_k10_AAL_s9 - w_k10_AAL_s96
                                 + 0.99 b_k10_AAL_s96 - 1.01 s_k10_AAL_s96  = 0
 w_k10_AAL_s97_return_constraint: 1.07876 w_k10_AAL_s9 - w_k10_AAL_s97
                                 + 0.99 b_k10_AAL_s97 - 1.01 s_k10_AAL_s97  = 0
 w_k10_AAL_s98_return_constraint: 0.967986 w_k10_AAL_s9 - w_k10_AAL_s98
                                 + 0.99 b_k10_AAL_s98 - 1.01 s_k10_AAL_s98  = 0
 w_k10_AAL_s99_return_constraint: 0.956497 w_k10_AAL_s9 - w_k10_AAL_s99
                                 + 0.99 b_k10_AAL_s99 - 1.01 s_k10_AAL_s99  = 0
 w_k10_PRU_s0_return_constraint: 1.07941 w_0_PRU - w_k10_PRU_s0
                                 + 0.99 b_k10_PRU_s0 - 1.01 s_k10_PRU_s0  = 0
 w_k10_PRU_s1_return_constraint: 0.908896 w_0_PRU - w_k10_PRU_s1
                                 + 0.99 b_k10_PRU_s1 - 1.01 s_k10_PRU_s1  = 0
 w_k10_PRU_s2_return_constraint: 1.06426 w_0_PRU - w_k10_PRU_s2
                                 + 0.99 b_k10_PRU_s2 - 1.01 s_k10_PRU_s2  = 0
 w_k10_PRU_s3_return_constraint: 1.12631 w_0_PRU - w_k10_PRU_s3
                                 + 0.99 b_k10_PRU_s3 - 1.01 s_k10_PRU_s3  = 0
 w_k10_PRU_s4_return_constraint: 0.96539 w_0_PRU - w_k10_PRU_s4
                                 + 0.99 b_k10_PRU_s4 - 1.01 s_k10_PRU_s4  = 0
 w_k10_PRU_s5_return_constraint: 0.848861 w_0_PRU - w_k10_PRU_s5
                                 + 0.99 b_k10_PRU_s5 - 1.01 s_k10_PRU_s5  = 0
 w_k10_PRU_s6_return_constraint: 0.974743 w_0_PRU - w_k10_PRU_s6
                                 + 0.99 b_k10_PRU_s6 - 1.01 s_k10_PRU_s6  = 0
 w_k10_PRU_s7_return_constraint: 1.17911 w_0_PRU - w_k10_PRU_s7
                                 + 0.99 b_k10_PRU_s7 - 1.01 s_k10_PRU_s7  = 0
 w_k10_PRU_s8_return_constraint: 1.01581 w_0_PRU - w_k10_PRU_s8
                                 + 0.99 b_k10_PRU_s8 - 1.01 s_k10_PRU_s8  = 0
 w_k10_PRU_s9_return_constraint: 0.91606 w_0_PRU - w_k10_PRU_s9
                                 + 0.99 b_k10_PRU_s9 - 1.01 s_k10_PRU_s9  = 0
 w_k10_PRU_s00_return_constraint: 1.01645 w_k10_PRU_s0 - w_k10_PRU_s00
                                 + 0.99 b_k10_PRU_s00 - 1.01 s_k10_PRU_s00  = 0
 w_k10_PRU_s01_return_constraint: 1.0804 w_k10_PRU_s0 - w_k10_PRU_s01
                                 + 0.99 b_k10_PRU_s01 - 1.01 s_k10_PRU_s01  = 0
 w_k10_PRU_s02_return_constraint: 0.909266 w_k10_PRU_s0 - w_k10_PRU_s02
                                 + 0.99 b_k10_PRU_s02 - 1.01 s_k10_PRU_s02  = 0
 w_k10_PRU_s03_return_constraint: 1.18322 w_k10_PRU_s0 - w_k10_PRU_s03
                                 + 0.99 b_k10_PRU_s03 - 1.01 s_k10_PRU_s03  = 0
 w_k10_PRU_s04_return_constraint: 0.939824 w_k10_PRU_s0 - w_k10_PRU_s04
                                 + 0.99 b_k10_PRU_s04 - 1.01 s_k10_PRU_s04  = 0
 w_k10_PRU_s05_return_constraint: 0.822526 w_k10_PRU_s0 - w_k10_PRU_s05
                                 + 0.99 b_k10_PRU_s05 - 1.01 s_k10_PRU_s05  = 0
 w_k10_PRU_s06_return_constraint: 1.28665 w_k10_PRU_s0 - w_k10_PRU_s06
                                 + 0.99 b_k10_PRU_s06 - 1.01 s_k10_PRU_s06  = 0
 w_k10_PRU_s07_return_constraint: 0.927106 w_k10_PRU_s0 - w_k10_PRU_s07
                                 + 0.99 b_k10_PRU_s07 - 1.01 s_k10_PRU_s07  = 0
 w_k10_PRU_s08_return_constraint: 1.042 w_k10_PRU_s0 - w_k10_PRU_s08
                                 + 0.99 b_k10_PRU_s08 - 1.01 s_k10_PRU_s08  = 0
 w_k10_PRU_s09_return_constraint: 1.14686 w_k10_PRU_s0 - w_k10_PRU_s09
                                 + 0.99 b_k10_PRU_s09 - 1.01 s_k10_PRU_s09  = 0
 w_k10_PRU_s10_return_constraint: 1.0212 w_k10_PRU_s1 - w_k10_PRU_s10
                                 + 0.99 b_k10_PRU_s10 - 1.01 s_k10_PRU_s10  = 0
 w_k10_PRU_s11_return_constraint: 1.15428 w_k10_PRU_s1 - w_k10_PRU_s11
                                 + 0.99 b_k10_PRU_s11 - 1.01 s_k10_PRU_s11  = 0
 w_k10_PRU_s12_return_constraint: 1.25808 w_k10_PRU_s1 - w_k10_PRU_s12
                                 + 0.99 b_k10_PRU_s12 - 1.01 s_k10_PRU_s12  = 0
 w_k10_PRU_s13_return_constraint: 0.892686 w_k10_PRU_s1 - w_k10_PRU_s13
                                 + 0.99 b_k10_PRU_s13 - 1.01 s_k10_PRU_s13  = 0
 w_k10_PRU_s14_return_constraint: 0.947114 w_k10_PRU_s1 - w_k10_PRU_s14
                                 + 0.99 b_k10_PRU_s14 - 1.01 s_k10_PRU_s14  = 0
 w_k10_PRU_s15_return_constraint: 1.14973 w_k10_PRU_s1 - w_k10_PRU_s15
                                 + 0.99 b_k10_PRU_s15 - 1.01 s_k10_PRU_s15  = 0
 w_k10_PRU_s16_return_constraint: 1.06714 w_k10_PRU_s1 - w_k10_PRU_s16
                                 + 0.99 b_k10_PRU_s16 - 1.01 s_k10_PRU_s16  = 0
 w_k10_PRU_s17_return_constraint: 0.839467 w_k10_PRU_s1 - w_k10_PRU_s17
                                 + 0.99 b_k10_PRU_s17 - 1.01 s_k10_PRU_s17  = 0
 w_k10_PRU_s18_return_constraint: 0.774374 w_k10_PRU_s1 - w_k10_PRU_s18
                                 + 0.99 b_k10_PRU_s18 - 1.01 s_k10_PRU_s18  = 0
 w_k10_PRU_s19_return_constraint: 1.20499 w_k10_PRU_s1 - w_k10_PRU_s19
                                 + 0.99 b_k10_PRU_s19 - 1.01 s_k10_PRU_s19  = 0
 w_k10_PRU_s20_return_constraint: 0.938684 w_k10_PRU_s2 - w_k10_PRU_s20
                                 + 0.99 b_k10_PRU_s20 - 1.01 s_k10_PRU_s20  = 0
 w_k10_PRU_s21_return_constraint: 1.0092 w_k10_PRU_s2 - w_k10_PRU_s21
                                 + 0.99 b_k10_PRU_s21 - 1.01 s_k10_PRU_s21  = 0
 w_k10_PRU_s22_return_constraint: 0.803037 w_k10_PRU_s2 - w_k10_PRU_s22
                                 + 0.99 b_k10_PRU_s22 - 1.01 s_k10_PRU_s22  = 0
 w_k10_PRU_s23_return_constraint: 1.07502 w_k10_PRU_s2 - w_k10_PRU_s23
                                 + 0.99 b_k10_PRU_s23 - 1.01 s_k10_PRU_s23  = 0
 w_k10_PRU_s24_return_constraint: 0.919502 w_k10_PRU_s2 - w_k10_PRU_s24
                                 + 0.99 b_k10_PRU_s24 - 1.01 s_k10_PRU_s24  = 0
 w_k10_PRU_s25_return_constraint: 0.952155 w_k10_PRU_s2 - w_k10_PRU_s25
                                 + 0.99 b_k10_PRU_s25 - 1.01 s_k10_PRU_s25  = 0
 w_k10_PRU_s26_return_constraint: 0.835948 w_k10_PRU_s2 - w_k10_PRU_s26
                                 + 0.99 b_k10_PRU_s26 - 1.01 s_k10_PRU_s26  = 0
 w_k10_PRU_s27_return_constraint: 0.863409 w_k10_PRU_s2 - w_k10_PRU_s27
                                 + 0.99 b_k10_PRU_s27 - 1.01 s_k10_PRU_s27  = 0
 w_k10_PRU_s28_return_constraint: 1.29125 w_k10_PRU_s2 - w_k10_PRU_s28
                                 + 0.99 b_k10_PRU_s28 - 1.01 s_k10_PRU_s28  = 0
 w_k10_PRU_s29_return_constraint: 1.11515 w_k10_PRU_s2 - w_k10_PRU_s29
                                 + 0.99 b_k10_PRU_s29 - 1.01 s_k10_PRU_s29  = 0
 w_k10_PRU_s30_return_constraint: 1.29328 w_k10_PRU_s3 - w_k10_PRU_s30
                                 + 0.99 b_k10_PRU_s30 - 1.01 s_k10_PRU_s30  = 0
 w_k10_PRU_s31_return_constraint: 0.891191 w_k10_PRU_s3 - w_k10_PRU_s31
                                 + 0.99 b_k10_PRU_s31 - 1.01 s_k10_PRU_s31  = 0
 w_k10_PRU_s32_return_constraint: 1.08626 w_k10_PRU_s3 - w_k10_PRU_s32
                                 + 0.99 b_k10_PRU_s32 - 1.01 s_k10_PRU_s32  = 0
 w_k10_PRU_s33_return_constraint: 1.01864 w_k10_PRU_s3 - w_k10_PRU_s33
                                 + 0.99 b_k10_PRU_s33 - 1.01 s_k10_PRU_s33  = 0
 w_k10_PRU_s34_return_constraint: 0.930864 w_k10_PRU_s3 - w_k10_PRU_s34
                                 + 0.99 b_k10_PRU_s34 - 1.01 s_k10_PRU_s34  = 0
 w_k10_PRU_s35_return_constraint: 1.02337 w_k10_PRU_s3 - w_k10_PRU_s35
                                 + 0.99 b_k10_PRU_s35 - 1.01 s_k10_PRU_s35  = 0
 w_k10_PRU_s36_return_constraint: 0.934002 w_k10_PRU_s3 - w_k10_PRU_s36
                                 + 0.99 b_k10_PRU_s36 - 1.01 s_k10_PRU_s36  = 0
 w_k10_PRU_s37_return_constraint: 0.878413 w_k10_PRU_s3 - w_k10_PRU_s37
                                 + 0.99 b_k10_PRU_s37 - 1.01 s_k10_PRU_s37  = 0
 w_k10_PRU_s38_return_constraint: 1.05967 w_k10_PRU_s3 - w_k10_PRU_s38
                                 + 0.99 b_k10_PRU_s38 - 1.01 s_k10_PRU_s38  = 0
 w_k10_PRU_s39_return_constraint: 0.909244 w_k10_PRU_s3 - w_k10_PRU_s39
                                 + 0.99 b_k10_PRU_s39 - 1.01 s_k10_PRU_s39  = 0
 w_k10_PRU_s40_return_constraint: 0.808883 w_k10_PRU_s4 - w_k10_PRU_s40
                                 + 0.99 b_k10_PRU_s40 - 1.01 s_k10_PRU_s40  = 0
 w_k10_PRU_s41_return_constraint: 0.878459 w_k10_PRU_s4 - w_k10_PRU_s41
                                 + 0.99 b_k10_PRU_s41 - 1.01 s_k10_PRU_s41  = 0
 w_k10_PRU_s42_return_constraint: 0.864699 w_k10_PRU_s4 - w_k10_PRU_s42
                                 + 0.99 b_k10_PRU_s42 - 1.01 s_k10_PRU_s42  = 0
 w_k10_PRU_s43_return_constraint: 0.889896 w_k10_PRU_s4 - w_k10_PRU_s43
                                 + 0.99 b_k10_PRU_s43 - 1.01 s_k10_PRU_s43  = 0
 w_k10_PRU_s44_return_constraint: 0.99718 w_k10_PRU_s4 - w_k10_PRU_s44
                                 + 0.99 b_k10_PRU_s44 - 1.01 s_k10_PRU_s44  = 0
 w_k10_PRU_s45_return_constraint: 1.03465 w_k10_PRU_s4 - w_k10_PRU_s45
                                 + 0.99 b_k10_PRU_s45 - 1.01 s_k10_PRU_s45  = 0
 w_k10_PRU_s46_return_constraint: 1.01605 w_k10_PRU_s4 - w_k10_PRU_s46
                                 + 0.99 b_k10_PRU_s46 - 1.01 s_k10_PRU_s46  = 0
 w_k10_PRU_s47_return_constraint: 1.19594 w_k10_PRU_s4 - w_k10_PRU_s47
                                 + 0.99 b_k10_PRU_s47 - 1.01 s_k10_PRU_s47  = 0
 w_k10_PRU_s48_return_constraint: 0.982539 w_k10_PRU_s4 - w_k10_PRU_s48
                                 + 0.99 b_k10_PRU_s48 - 1.01 s_k10_PRU_s48  = 0
 w_k10_PRU_s49_return_constraint: 0.94262 w_k10_PRU_s4 - w_k10_PRU_s49
                                 + 0.99 b_k10_PRU_s49 - 1.01 s_k10_PRU_s49  = 0
 w_k10_PRU_s50_return_constraint: 1.15722 w_k10_PRU_s5 - w_k10_PRU_s50
                                 + 0.99 b_k10_PRU_s50 - 1.01 s_k10_PRU_s50  = 0
 w_k10_PRU_s51_return_constraint: 0.89159 w_k10_PRU_s5 - w_k10_PRU_s51
                                 + 0.99 b_k10_PRU_s51 - 1.01 s_k10_PRU_s51  = 0
 w_k10_PRU_s52_return_constraint: 1.1016 w_k10_PRU_s5 - w_k10_PRU_s52
                                 + 0.99 b_k10_PRU_s52 - 1.01 s_k10_PRU_s52  = 0
 w_k10_PRU_s53_return_constraint: 0.955962 w_k10_PRU_s5 - w_k10_PRU_s53
                                 + 0.99 b_k10_PRU_s53 - 1.01 s_k10_PRU_s53  = 0
 w_k10_PRU_s54_return_constraint: 1.02902 w_k10_PRU_s5 - w_k10_PRU_s54
                                 + 0.99 b_k10_PRU_s54 - 1.01 s_k10_PRU_s54  = 0
 w_k10_PRU_s55_return_constraint: 0.869703 w_k10_PRU_s5 - w_k10_PRU_s55
                                 + 0.99 b_k10_PRU_s55 - 1.01 s_k10_PRU_s55  = 0
 w_k10_PRU_s56_return_constraint: 1.1338 w_k10_PRU_s5 - w_k10_PRU_s56
                                 + 0.99 b_k10_PRU_s56 - 1.01 s_k10_PRU_s56  = 0
 w_k10_PRU_s57_return_constraint: 0.808934 w_k10_PRU_s5 - w_k10_PRU_s57
                                 + 0.99 b_k10_PRU_s57 - 1.01 s_k10_PRU_s57  = 0
 w_k10_PRU_s58_return_constraint: 0.809677 w_k10_PRU_s5 - w_k10_PRU_s58
                                 + 0.99 b_k10_PRU_s58 - 1.01 s_k10_PRU_s58  = 0
 w_k10_PRU_s59_return_constraint: 1.09759 w_k10_PRU_s5 - w_k10_PRU_s59
                                 + 0.99 b_k10_PRU_s59 - 1.01 s_k10_PRU_s59  = 0
 w_k10_PRU_s60_return_constraint: 0.964279 w_k10_PRU_s6 - w_k10_PRU_s60
                                 + 0.99 b_k10_PRU_s60 - 1.01 s_k10_PRU_s60  = 0
 w_k10_PRU_s61_return_constraint: 1.15028 w_k10_PRU_s6 - w_k10_PRU_s61
                                 + 0.99 b_k10_PRU_s61 - 1.01 s_k10_PRU_s61  = 0
 w_k10_PRU_s62_return_constraint: 1.12188 w_k10_PRU_s6 - w_k10_PRU_s62
                                 + 0.99 b_k10_PRU_s62 - 1.01 s_k10_PRU_s62  = 0
 w_k10_PRU_s63_return_constraint: 0.999074 w_k10_PRU_s6 - w_k10_PRU_s63
                                 + 0.99 b_k10_PRU_s63 - 1.01 s_k10_PRU_s63  = 0
 w_k10_PRU_s64_return_constraint: 0.944005 w_k10_PRU_s6 - w_k10_PRU_s64
                                 + 0.99 b_k10_PRU_s64 - 1.01 s_k10_PRU_s64  = 0
 w_k10_PRU_s65_return_constraint: 0.952091 w_k10_PRU_s6 - w_k10_PRU_s65
                                 + 0.99 b_k10_PRU_s65 - 1.01 s_k10_PRU_s65  = 0
 w_k10_PRU_s66_return_constraint: 1.0751 w_k10_PRU_s6 - w_k10_PRU_s66
                                 + 0.99 b_k10_PRU_s66 - 1.01 s_k10_PRU_s66  = 0
 w_k10_PRU_s67_return_constraint: 0.914893 w_k10_PRU_s6 - w_k10_PRU_s67
                                 + 0.99 b_k10_PRU_s67 - 1.01 s_k10_PRU_s67  = 0
 w_k10_PRU_s68_return_constraint: 1.17264 w_k10_PRU_s6 - w_k10_PRU_s68
                                 + 0.99 b_k10_PRU_s68 - 1.01 s_k10_PRU_s68  = 0
 w_k10_PRU_s69_return_constraint: 0.890839 w_k10_PRU_s6 - w_k10_PRU_s69
                                 + 0.99 b_k10_PRU_s69 - 1.01 s_k10_PRU_s69  = 0
 w_k10_PRU_s70_return_constraint: 0.983787 w_k10_PRU_s7 - w_k10_PRU_s70
                                 + 0.99 b_k10_PRU_s70 - 1.01 s_k10_PRU_s70  = 0
 w_k10_PRU_s71_return_constraint: 1.03896 w_k10_PRU_s7 - w_k10_PRU_s71
                                 + 0.99 b_k10_PRU_s71 - 1.01 s_k10_PRU_s71  = 0
 w_k10_PRU_s72_return_constraint: 1.04904 w_k10_PRU_s7 - w_k10_PRU_s72
                                 + 0.99 b_k10_PRU_s72 - 1.01 s_k10_PRU_s72  = 0
 w_k10_PRU_s73_return_constraint: 1.18518 w_k10_PRU_s7 - w_k10_PRU_s73
                                 + 0.99 b_k10_PRU_s73 - 1.01 s_k10_PRU_s73  = 0
 w_k10_PRU_s74_return_constraint: 0.971776 w_k10_PRU_s7 - w_k10_PRU_s74
                                 + 0.99 b_k10_PRU_s74 - 1.01 s_k10_PRU_s74  = 0
 w_k10_PRU_s75_return_constraint: 1.022 w_k10_PRU_s7 - w_k10_PRU_s75
                                 + 0.99 b_k10_PRU_s75 - 1.01 s_k10_PRU_s75  = 0
 w_k10_PRU_s76_return_constraint: 0.953638 w_k10_PRU_s7 - w_k10_PRU_s76
                                 + 0.99 b_k10_PRU_s76 - 1.01 s_k10_PRU_s76  = 0
 w_k10_PRU_s77_return_constraint: 1.13441 w_k10_PRU_s7 - w_k10_PRU_s77
                                 + 0.99 b_k10_PRU_s77 - 1.01 s_k10_PRU_s77  = 0
 w_k10_PRU_s78_return_constraint: 0.748236 w_k10_PRU_s7 - w_k10_PRU_s78
                                 + 0.99 b_k10_PRU_s78 - 1.01 s_k10_PRU_s78  = 0
 w_k10_PRU_s79_return_constraint: 0.885153 w_k10_PRU_s7 - w_k10_PRU_s79
                                 + 0.99 b_k10_PRU_s79 - 1.01 s_k10_PRU_s79  = 0
 w_k10_PRU_s80_return_constraint: 0.995406 w_k10_PRU_s8 - w_k10_PRU_s80
                                 + 0.99 b_k10_PRU_s80 - 1.01 s_k10_PRU_s80  = 0
 w_k10_PRU_s81_return_constraint: 0.94842 w_k10_PRU_s8 - w_k10_PRU_s81
                                 + 0.99 b_k10_PRU_s81 - 1.01 s_k10_PRU_s81  = 0
 w_k10_PRU_s82_return_constraint: 0.936175 w_k10_PRU_s8 - w_k10_PRU_s82
                                 + 0.99 b_k10_PRU_s82 - 1.01 s_k10_PRU_s82  = 0
 w_k10_PRU_s83_return_constraint: 1.21518 w_k10_PRU_s8 - w_k10_PRU_s83
                                 + 0.99 b_k10_PRU_s83 - 1.01 s_k10_PRU_s83  = 0
 w_k10_PRU_s84_return_constraint: 0.895093 w_k10_PRU_s8 - w_k10_PRU_s84
                                 + 0.99 b_k10_PRU_s84 - 1.01 s_k10_PRU_s84  = 0
 w_k10_PRU_s85_return_constraint: 0.847904 w_k10_PRU_s8 - w_k10_PRU_s85
                                 + 0.99 b_k10_PRU_s85 - 1.01 s_k10_PRU_s85  = 0
 w_k10_PRU_s86_return_constraint: 0.951531 w_k10_PRU_s8 - w_k10_PRU_s86
                                 + 0.99 b_k10_PRU_s86 - 1.01 s_k10_PRU_s86  = 0
 w_k10_PRU_s87_return_constraint: 1.1115 w_k10_PRU_s8 - w_k10_PRU_s87
                                 + 0.99 b_k10_PRU_s87 - 1.01 s_k10_PRU_s87  = 0
 w_k10_PRU_s88_return_constraint: 1.02522 w_k10_PRU_s8 - w_k10_PRU_s88
                                 + 0.99 b_k10_PRU_s88 - 1.01 s_k10_PRU_s88  = 0
 w_k10_PRU_s89_return_constraint: 0.779891 w_k10_PRU_s8 - w_k10_PRU_s89
                                 + 0.99 b_k10_PRU_s89 - 1.01 s_k10_PRU_s89  = 0
 w_k10_PRU_s90_return_constraint: 1.22501 w_k10_PRU_s9 - w_k10_PRU_s90
                                 + 0.99 b_k10_PRU_s90 - 1.01 s_k10_PRU_s90  = 0
 w_k10_PRU_s91_return_constraint: 0.99343 w_k10_PRU_s9 - w_k10_PRU_s91
                                 + 0.99 b_k10_PRU_s91 - 1.01 s_k10_PRU_s91  = 0
 w_k10_PRU_s92_return_constraint: 1.04107 w_k10_PRU_s9 - w_k10_PRU_s92
                                 + 0.99 b_k10_PRU_s92 - 1.01 s_k10_PRU_s92  = 0
 w_k10_PRU_s93_return_constraint: 0.917177 w_k10_PRU_s9 - w_k10_PRU_s93
                                 + 0.99 b_k10_PRU_s93 - 1.01 s_k10_PRU_s93  = 0
 w_k10_PRU_s94_return_constraint: 1.17874 w_k10_PRU_s9 - w_k10_PRU_s94
                                 + 0.99 b_k10_PRU_s94 - 1.01 s_k10_PRU_s94  = 0
 w_k10_PRU_s95_return_constraint: 0.904899 w_k10_PRU_s9 - w_k10_PRU_s95
                                 + 0.99 b_k10_PRU_s95 - 1.01 s_k10_PRU_s95  = 0
 w_k10_PRU_s96_return_constraint: 0.844172 w_k10_PRU_s9 - w_k10_PRU_s96
                                 + 0.99 b_k10_PRU_s96 - 1.01 s_k10_PRU_s96  = 0
 w_k10_PRU_s97_return_constraint: 0.978121 w_k10_PRU_s9 - w_k10_PRU_s97
                                 + 0.99 b_k10_PRU_s97 - 1.01 s_k10_PRU_s97  = 0
 w_k10_PRU_s98_return_constraint: 0.894989 w_k10_PRU_s9 - w_k10_PRU_s98
                                 + 0.99 b_k10_PRU_s98 - 1.01 s_k10_PRU_s98  = 0
 w_k10_PRU_s99_return_constraint: 1.10139 w_k10_PRU_s9 - w_k10_PRU_s99
                                 + 0.99 b_k10_PRU_s99 - 1.01 s_k10_PRU_s99  = 0
 k10_s0_balance_constraint:      b_k10_HSBC_s0 + b_k10_VOD_s0 + b_k10_BP_s0
                                 + b_k10_GSK_s0 + b_k10_AZN_s0 + b_k10_RIO_s0
                                 + b_k10_BG_s0 + b_k10_TSCO_s0 + b_k10_AAL_s0
                                 + b_k10_PRU_s0 - s_k10_HSBC_s0 - s_k10_VOD_s0
                                 - s_k10_BP_s0 - s_k10_GSK_s0 - s_k10_AZN_s0
                                 - s_k10_RIO_s0 - s_k10_BG_s0 - s_k10_TSCO_s0
                                 - s_k10_AAL_s0 - s_k10_PRU_s0  = 0
 k10_s1_balance_constraint:      b_k10_HSBC_s1 + b_k10_VOD_s1 + b_k10_BP_s1
                                 + b_k10_GSK_s1 + b_k10_AZN_s1 + b_k10_RIO_s1
                                 + b_k10_BG_s1 + b_k10_TSCO_s1 + b_k10_AAL_s1
                                 + b_k10_PRU_s1 - s_k10_HSBC_s1 - s_k10_VOD_s1
                                 - s_k10_BP_s1 - s_k10_GSK_s1 - s_k10_AZN_s1
                                 - s_k10_RIO_s1 - s_k10_BG_s1 - s_k10_TSCO_s1
                                 - s_k10_AAL_s1 - s_k10_PRU_s1  = 0
 k10_s2_balance_constraint:      b_k10_HSBC_s2 + b_k10_VOD_s2 + b_k10_BP_s2
                                 + b_k10_GSK_s2 + b_k10_AZN_s2 + b_k10_RIO_s2
                                 + b_k10_BG_s2 + b_k10_TSCO_s2 + b_k10_AAL_s2
                                 + b_k10_PRU_s2 - s_k10_HSBC_s2 - s_k10_VOD_s2
                                 - s_k10_BP_s2 - s_k10_GSK_s2 - s_k10_AZN_s2
                                 - s_k10_RIO_s2 - s_k10_BG_s2 - s_k10_TSCO_s2
                                 - s_k10_AAL_s2 - s_k10_PRU_s2  = 0
 k10_s3_balance_constraint:      b_k10_HSBC_s3 + b_k10_VOD_s3 + b_k10_BP_s3
                                 + b_k10_GSK_s3 + b_k10_AZN_s3 + b_k10_RIO_s3
                                 + b_k10_BG_s3 + b_k10_TSCO_s3 + b_k10_AAL_s3
                                 + b_k10_PRU_s3 - s_k10_HSBC_s3 - s_k10_VOD_s3
                                 - s_k10_BP_s3 - s_k10_GSK_s3 - s_k10_AZN_s3
                                 - s_k10_RIO_s3 - s_k10_BG_s3 - s_k10_TSCO_s3
                                 - s_k10_AAL_s3 - s_k10_PRU_s3  = 0
 k10_s4_balance_constraint:      b_k10_HSBC_s4 + b_k10_VOD_s4 + b_k10_BP_s4
                                 + b_k10_GSK_s4 + b_k10_AZN_s4 + b_k10_RIO_s4
                                 + b_k10_BG_s4 + b_k10_TSCO_s4 + b_k10_AAL_s4
                                 + b_k10_PRU_s4 - s_k10_HSBC_s4 - s_k10_VOD_s4
                                 - s_k10_BP_s4 - s_k10_GSK_s4 - s_k10_AZN_s4
                                 - s_k10_RIO_s4 - s_k10_BG_s4 - s_k10_TSCO_s4
                                 - s_k10_AAL_s4 - s_k10_PRU_s4  = 0
 k10_s5_balance_constraint:      b_k10_HSBC_s5 + b_k10_VOD_s5 + b_k10_BP_s5
                                 + b_k10_GSK_s5 + b_k10_AZN_s5 + b_k10_RIO_s5
                                 + b_k10_BG_s5 + b_k10_TSCO_s5 + b_k10_AAL_s5
                                 + b_k10_PRU_s5 - s_k10_HSBC_s5 - s_k10_VOD_s5
                                 - s_k10_BP_s5 - s_k10_GSK_s5 - s_k10_AZN_s5
                                 - s_k10_RIO_s5 - s_k10_BG_s5 - s_k10_TSCO_s5
                                 - s_k10_AAL_s5 - s_k10_PRU_s5  = 0
 k10_s6_balance_constraint:      b_k10_HSBC_s6 + b_k10_VOD_s6 + b_k10_BP_s6
                                 + b_k10_GSK_s6 + b_k10_AZN_s6 + b_k10_RIO_s6
                                 + b_k10_BG_s6 + b_k10_TSCO_s6 + b_k10_AAL_s6
                                 + b_k10_PRU_s6 - s_k10_HSBC_s6 - s_k10_VOD_s6
                                 - s_k10_BP_s6 - s_k10_GSK_s6 - s_k10_AZN_s6
                                 - s_k10_RIO_s6 - s_k10_BG_s6 - s_k10_TSCO_s6
                                 - s_k10_AAL_s6 - s_k10_PRU_s6  = 0
 k10_s7_balance_constraint:      b_k10_HSBC_s7 + b_k10_VOD_s7 + b_k10_BP_s7
                                 + b_k10_GSK_s7 + b_k10_AZN_s7 + b_k10_RIO_s7
                                 + b_k10_BG_s7 + b_k10_TSCO_s7 + b_k10_AAL_s7
                                 + b_k10_PRU_s7 - s_k10_HSBC_s7 - s_k10_VOD_s7
                                 - s_k10_BP_s7 - s_k10_GSK_s7 - s_k10_AZN_s7
                                 - s_k10_RIO_s7 - s_k10_BG_s7 - s_k10_TSCO_s7
                                 - s_k10_AAL_s7 - s_k10_PRU_s7  = 0
 k10_s8_balance_constraint:      b_k10_HSBC_s8 + b_k10_VOD_s8 + b_k10_BP_s8
                                 + b_k10_GSK_s8 + b_k10_AZN_s8 + b_k10_RIO_s8
                                 + b_k10_BG_s8 + b_k10_TSCO_s8 + b_k10_AAL_s8
                                 + b_k10_PRU_s8 - s_k10_HSBC_s8 - s_k10_VOD_s8
                                 - s_k10_BP_s8 - s_k10_GSK_s8 - s_k10_AZN_s8
                                 - s_k10_RIO_s8 - s_k10_BG_s8 - s_k10_TSCO_s8
                                 - s_k10_AAL_s8 - s_k10_PRU_s8  = 0
 k10_s9_balance_constraint:      b_k10_HSBC_s9 + b_k10_VOD_s9 + b_k10_BP_s9
                                 + b_k10_GSK_s9 + b_k10_AZN_s9 + b_k10_RIO_s9
                                 + b_k10_BG_s9 + b_k10_TSCO_s9 + b_k10_AAL_s9
                                 + b_k10_PRU_s9 - s_k10_HSBC_s9 - s_k10_VOD_s9
                                 - s_k10_BP_s9 - s_k10_GSK_s9 - s_k10_AZN_s9
                                 - s_k10_RIO_s9 - s_k10_BG_s9 - s_k10_TSCO_s9
                                 - s_k10_AAL_s9 - s_k10_PRU_s9  = 0
 k10_s00_balance_constraint:     b_k10_HSBC_s00 + b_k10_VOD_s00 + b_k10_BP_s00
                                 + b_k10_GSK_s00 + b_k10_AZN_s00
                                 + b_k10_RIO_s00 + b_k10_BG_s00
                                 + b_k10_TSCO_s00 + b_k10_AAL_s00
                                 + b_k10_PRU_s00 - s_k10_HSBC_s00
                                 - s_k10_VOD_s00 - s_k10_BP_s00 - s_k10_GSK_s00
                                 - s_k10_AZN_s00 - s_k10_RIO_s00 - s_k10_BG_s00
                                 - s_k10_TSCO_s00 - s_k10_AAL_s00
                                 - s_k10_PRU_s00  = 0
 k10_s01_balance_constraint:     b_k10_HSBC_s01 + b_k10_VOD_s01 + b_k10_BP_s01
                                 + b_k10_GSK_s01 + b_k10_AZN_s01
                                 + b_k10_RIO_s01 + b_k10_BG_s01
                                 + b_k10_TSCO_s01 + b_k10_AAL_s01
                                 + b_k10_PRU_s01 - s_k10_HSBC_s01
                                 - s_k10_VOD_s01 - s_k10_BP_s01 - s_k10_GSK_s01
                                 - s_k10_AZN_s01 - s_k10_RIO_s01 - s_k10_BG_s01
                                 - s_k10_TSCO_s01 - s_k10_AAL_s01
                                 - s_k10_PRU_s01  = 0
 k10_s02_balance_constraint:     b_k10_HSBC_s02 + b_k10_VOD_s02 + b_k10_BP_s02
                                 + b_k10_GSK_s02 + b_k10_AZN_s02
                                 + b_k10_RIO_s02 + b_k10_BG_s02
                                 + b_k10_TSCO_s02 + b_k10_AAL_s02
                                 + b_k10_PRU_s02 - s_k10_HSBC_s02
                                 - s_k10_VOD_s02 - s_k10_BP_s02 - s_k10_GSK_s02
                                 - s_k10_AZN_s02 - s_k10_RIO_s02 - s_k10_BG_s02
                                 - s_k10_TSCO_s02 - s_k10_AAL_s02
                                 - s_k10_PRU_s02  = 0
 k10_s03_balance_constraint:     b_k10_HSBC_s03 + b_k10_VOD_s03 + b_k10_BP_s03
                                 + b_k10_GSK_s03 + b_k10_AZN_s03
                                 + b_k10_RIO_s03 + b_k10_BG_s03
                                 + b_k10_TSCO_s03 + b_k10_AAL_s03
                                 + b_k10_PRU_s03 - s_k10_HSBC_s03
                                 - s_k10_VOD_s03 - s_k10_BP_s03 - s_k10_GSK_s03
                                 - s_k10_AZN_s03 - s_k10_RIO_s03 - s_k10_BG_s03
                                 - s_k10_TSCO_s03 - s_k10_AAL_s03
                                 - s_k10_PRU_s03  = 0
 k10_s04_balance_constraint:     b_k10_HSBC_s04 + b_k10_VOD_s04 + b_k10_BP_s04
                                 + b_k10_GSK_s04 + b_k10_AZN_s04
                                 + b_k10_RIO_s04 + b_k10_BG_s04
                                 + b_k10_TSCO_s04 + b_k10_AAL_s04
                                 + b_k10_PRU_s04 - s_k10_HSBC_s04
                                 - s_k10_VOD_s04 - s_k10_BP_s04 - s_k10_GSK_s04
                                 - s_k10_AZN_s04 - s_k10_RIO_s04 - s_k10_BG_s04
                                 - s_k10_TSCO_s04 - s_k10_AAL_s04
                                 - s_k10_PRU_s04  = 0
 k10_s05_balance_constraint:     b_k10_HSBC_s05 + b_k10_VOD_s05 + b_k10_BP_s05
                                 + b_k10_GSK_s05 + b_k10_AZN_s05
                                 + b_k10_RIO_s05 + b_k10_BG_s05
                                 + b_k10_TSCO_s05 + b_k10_AAL_s05
                                 + b_k10_PRU_s05 - s_k10_HSBC_s05
                                 - s_k10_VOD_s05 - s_k10_BP_s05 - s_k10_GSK_s05
                                 - s_k10_AZN_s05 - s_k10_RIO_s05 - s_k10_BG_s05
                                 - s_k10_TSCO_s05 - s_k10_AAL_s05
                                 - s_k10_PRU_s05  = 0
 k10_s06_balance_constraint:     b_k10_HSBC_s06 + b_k10_VOD_s06 + b_k10_BP_s06
                                 + b_k10_GSK_s06 + b_k10_AZN_s06
                                 + b_k10_RIO_s06 + b_k10_BG_s06
                                 + b_k10_TSCO_s06 + b_k10_AAL_s06
                                 + b_k10_PRU_s06 - s_k10_HSBC_s06
                                 - s_k10_VOD_s06 - s_k10_BP_s06 - s_k10_GSK_s06
                                 - s_k10_AZN_s06 - s_k10_RIO_s06 - s_k10_BG_s06
                                 - s_k10_TSCO_s06 - s_k10_AAL_s06
                                 - s_k10_PRU_s06  = 0
 k10_s07_balance_constraint:     b_k10_HSBC_s07 + b_k10_VOD_s07 + b_k10_BP_s07
                                 + b_k10_GSK_s07 + b_k10_AZN_s07
                                 + b_k10_RIO_s07 + b_k10_BG_s07
                                 + b_k10_TSCO_s07 + b_k10_AAL_s07
                                 + b_k10_PRU_s07 - s_k10_HSBC_s07
                                 - s_k10_VOD_s07 - s_k10_BP_s07 - s_k10_GSK_s07
                                 - s_k10_AZN_s07 - s_k10_RIO_s07 - s_k10_BG_s07
                                 - s_k10_TSCO_s07 - s_k10_AAL_s07
                                 - s_k10_PRU_s07  = 0
 k10_s08_balance_constraint:     b_k10_HSBC_s08 + b_k10_VOD_s08 + b_k10_BP_s08
                                 + b_k10_GSK_s08 + b_k10_AZN_s08
                                 + b_k10_RIO_s08 + b_k10_BG_s08
                                 + b_k10_TSCO_s08 + b_k10_AAL_s08
                                 + b_k10_PRU_s08 - s_k10_HSBC_s08
                                 - s_k10_VOD_s08 - s_k10_BP_s08 - s_k10_GSK_s08
                                 - s_k10_AZN_s08 - s_k10_RIO_s08 - s_k10_BG_s08
                                 - s_k10_TSCO_s08 - s_k10_AAL_s08
                                 - s_k10_PRU_s08  = 0
 k10_s09_balance_constraint:     b_k10_HSBC_s09 + b_k10_VOD_s09 + b_k10_BP_s09
                                 + b_k10_GSK_s09 + b_k10_AZN_s09
                                 + b_k10_RIO_s09 + b_k10_BG_s09
                                 + b_k10_TSCO_s09 + b_k10_AAL_s09
                                 + b_k10_PRU_s09 - s_k10_HSBC_s09
                                 - s_k10_VOD_s09 - s_k10_BP_s09 - s_k10_GSK_s09
                                 - s_k10_AZN_s09 - s_k10_RIO_s09 - s_k10_BG_s09
                                 - s_k10_TSCO_s09 - s_k10_AAL_s09
                                 - s_k10_PRU_s09  = 0
 k10_s10_balance_constraint:     b_k10_HSBC_s10 + b_k10_VOD_s10 + b_k10_BP_s10
                                 + b_k10_GSK_s10 + b_k10_AZN_s10
                                 + b_k10_RIO_s10 + b_k10_BG_s10
                                 + b_k10_TSCO_s10 + b_k10_AAL_s10
                                 + b_k10_PRU_s10 - s_k10_HSBC_s10
                                 - s_k10_VOD_s10 - s_k10_BP_s10 - s_k10_GSK_s10
                                 - s_k10_AZN_s10 - s_k10_RIO_s10 - s_k10_BG_s10
                                 - s_k10_TSCO_s10 - s_k10_AAL_s10
                                 - s_k10_PRU_s10  = 0
 k10_s11_balance_constraint:     b_k10_HSBC_s11 + b_k10_VOD_s11 + b_k10_BP_s11
                                 + b_k10_GSK_s11 + b_k10_AZN_s11
                                 + b_k10_RIO_s11 + b_k10_BG_s11
                                 + b_k10_TSCO_s11 + b_k10_AAL_s11
                                 + b_k10_PRU_s11 - s_k10_HSBC_s11
                                 - s_k10_VOD_s11 - s_k10_BP_s11 - s_k10_GSK_s11
                                 - s_k10_AZN_s11 - s_k10_RIO_s11 - s_k10_BG_s11
                                 - s_k10_TSCO_s11 - s_k10_AAL_s11
                                 - s_k10_PRU_s11  = 0
 k10_s12_balance_constraint:     b_k10_HSBC_s12 + b_k10_VOD_s12 + b_k10_BP_s12
                                 + b_k10_GSK_s12 + b_k10_AZN_s12
                                 + b_k10_RIO_s12 + b_k10_BG_s12
                                 + b_k10_TSCO_s12 + b_k10_AAL_s12
                                 + b_k10_PRU_s12 - s_k10_HSBC_s12
                                 - s_k10_VOD_s12 - s_k10_BP_s12 - s_k10_GSK_s12
                                 - s_k10_AZN_s12 - s_k10_RIO_s12 - s_k10_BG_s12
                                 - s_k10_TSCO_s12 - s_k10_AAL_s12
                                 - s_k10_PRU_s12  = 0
 k10_s13_balance_constraint:     b_k10_HSBC_s13 + b_k10_VOD_s13 + b_k10_BP_s13
                                 + b_k10_GSK_s13 + b_k10_AZN_s13
                                 + b_k10_RIO_s13 + b_k10_BG_s13
                                 + b_k10_TSCO_s13 + b_k10_AAL_s13
                                 + b_k10_PRU_s13 - s_k10_HSBC_s13
                                 - s_k10_VOD_s13 - s_k10_BP_s13 - s_k10_GSK_s13
                                 - s_k10_AZN_s13 - s_k10_RIO_s13 - s_k10_BG_s13
                                 - s_k10_TSCO_s13 - s_k10_AAL_s13
                                 - s_k10_PRU_s13  = 0
 k10_s14_balance_constraint:     b_k10_HSBC_s14 + b_k10_VOD_s14 + b_k10_BP_s14
                                 + b_k10_GSK_s14 + b_k10_AZN_s14
                                 + b_k10_RIO_s14 + b_k10_BG_s14
                                 + b_k10_TSCO_s14 + b_k10_AAL_s14
                                 + b_k10_PRU_s14 - s_k10_HSBC_s14
                                 - s_k10_VOD_s14 - s_k10_BP_s14 - s_k10_GSK_s14
                                 - s_k10_AZN_s14 - s_k10_RIO_s14 - s_k10_BG_s14
                                 - s_k10_TSCO_s14 - s_k10_AAL_s14
                                 - s_k10_PRU_s14  = 0
 k10_s15_balance_constraint:     b_k10_HSBC_s15 + b_k10_VOD_s15 + b_k10_BP_s15
                                 + b_k10_GSK_s15 + b_k10_AZN_s15
                                 + b_k10_RIO_s15 + b_k10_BG_s15
                                 + b_k10_TSCO_s15 + b_k10_AAL_s15
                                 + b_k10_PRU_s15 - s_k10_HSBC_s15
                                 - s_k10_VOD_s15 - s_k10_BP_s15 - s_k10_GSK_s15
                                 - s_k10_AZN_s15 - s_k10_RIO_s15 - s_k10_BG_s15
                                 - s_k10_TSCO_s15 - s_k10_AAL_s15
                                 - s_k10_PRU_s15  = 0
 k10_s16_balance_constraint:     b_k10_HSBC_s16 + b_k10_VOD_s16 + b_k10_BP_s16
                                 + b_k10_GSK_s16 + b_k10_AZN_s16
                                 + b_k10_RIO_s16 + b_k10_BG_s16
                                 + b_k10_TSCO_s16 + b_k10_AAL_s16
                                 + b_k10_PRU_s16 - s_k10_HSBC_s16
                                 - s_k10_VOD_s16 - s_k10_BP_s16 - s_k10_GSK_s16
                                 - s_k10_AZN_s16 - s_k10_RIO_s16 - s_k10_BG_s16
                                 - s_k10_TSCO_s16 - s_k10_AAL_s16
                                 - s_k10_PRU_s16  = 0
 k10_s17_balance_constraint:     b_k10_HSBC_s17 + b_k10_VOD_s17 + b_k10_BP_s17
                                 + b_k10_GSK_s17 + b_k10_AZN_s17
                                 + b_k10_RIO_s17 + b_k10_BG_s17
                                 + b_k10_TSCO_s17 + b_k10_AAL_s17
                                 + b_k10_PRU_s17 - s_k10_HSBC_s17
                                 - s_k10_VOD_s17 - s_k10_BP_s17 - s_k10_GSK_s17
                                 - s_k10_AZN_s17 - s_k10_RIO_s17 - s_k10_BG_s17
                                 - s_k10_TSCO_s17 - s_k10_AAL_s17
                                 - s_k10_PRU_s17  = 0
 k10_s18_balance_constraint:     b_k10_HSBC_s18 + b_k10_VOD_s18 + b_k10_BP_s18
                                 + b_k10_GSK_s18 + b_k10_AZN_s18
                                 + b_k10_RIO_s18 + b_k10_BG_s18
                                 + b_k10_TSCO_s18 + b_k10_AAL_s18
                                 + b_k10_PRU_s18 - s_k10_HSBC_s18
                                 - s_k10_VOD_s18 - s_k10_BP_s18 - s_k10_GSK_s18
                                 - s_k10_AZN_s18 - s_k10_RIO_s18 - s_k10_BG_s18
                                 - s_k10_TSCO_s18 - s_k10_AAL_s18
                                 - s_k10_PRU_s18  = 0
 k10_s19_balance_constraint:     b_k10_HSBC_s19 + b_k10_VOD_s19 + b_k10_BP_s19
                                 + b_k10_GSK_s19 + b_k10_AZN_s19
                                 + b_k10_RIO_s19 + b_k10_BG_s19
                                 + b_k10_TSCO_s19 + b_k10_AAL_s19
                                 + b_k10_PRU_s19 - s_k10_HSBC_s19
                                 - s_k10_VOD_s19 - s_k10_BP_s19 - s_k10_GSK_s19
                                 - s_k10_AZN_s19 - s_k10_RIO_s19 - s_k10_BG_s19
                                 - s_k10_TSCO_s19 - s_k10_AAL_s19
                                 - s_k10_PRU_s19  = 0
 k10_s20_balance_constraint:     b_k10_HSBC_s20 + b_k10_VOD_s20 + b_k10_BP_s20
                                 + b_k10_GSK_s20 + b_k10_AZN_s20
                                 + b_k10_RIO_s20 + b_k10_BG_s20
                                 + b_k10_TSCO_s20 + b_k10_AAL_s20
                                 + b_k10_PRU_s20 - s_k10_HSBC_s20
                                 - s_k10_VOD_s20 - s_k10_BP_s20 - s_k10_GSK_s20
                                 - s_k10_AZN_s20 - s_k10_RIO_s20 - s_k10_BG_s20
                                 - s_k10_TSCO_s20 - s_k10_AAL_s20
                                 - s_k10_PRU_s20  = 0
 k10_s21_balance_constraint:     b_k10_HSBC_s21 + b_k10_VOD_s21 + b_k10_BP_s21
                                 + b_k10_GSK_s21 + b_k10_AZN_s21
                                 + b_k10_RIO_s21 + b_k10_BG_s21
                                 + b_k10_TSCO_s21 + b_k10_AAL_s21
                                 + b_k10_PRU_s21 - s_k10_HSBC_s21
                                 - s_k10_VOD_s21 - s_k10_BP_s21 - s_k10_GSK_s21
                                 - s_k10_AZN_s21 - s_k10_RIO_s21 - s_k10_BG_s21
                                 - s_k10_TSCO_s21 - s_k10_AAL_s21
                                 - s_k10_PRU_s21  = 0
 k10_s22_balance_constraint:     b_k10_HSBC_s22 + b_k10_VOD_s22 + b_k10_BP_s22
                                 + b_k10_GSK_s22 + b_k10_AZN_s22
                                 + b_k10_RIO_s22 + b_k10_BG_s22
                                 + b_k10_TSCO_s22 + b_k10_AAL_s22
                                 + b_k10_PRU_s22 - s_k10_HSBC_s22
                                 - s_k10_VOD_s22 - s_k10_BP_s22 - s_k10_GSK_s22
                                 - s_k10_AZN_s22 - s_k10_RIO_s22 - s_k10_BG_s22
                                 - s_k10_TSCO_s22 - s_k10_AAL_s22
                                 - s_k10_PRU_s22  = 0
 k10_s23_balance_constraint:     b_k10_HSBC_s23 + b_k10_VOD_s23 + b_k10_BP_s23
                                 + b_k10_GSK_s23 + b_k10_AZN_s23
                                 + b_k10_RIO_s23 + b_k10_BG_s23
                                 + b_k10_TSCO_s23 + b_k10_AAL_s23
                                 + b_k10_PRU_s23 - s_k10_HSBC_s23
                                 - s_k10_VOD_s23 - s_k10_BP_s23 - s_k10_GSK_s23
                                 - s_k10_AZN_s23 - s_k10_RIO_s23 - s_k10_BG_s23
                                 - s_k10_TSCO_s23 - s_k10_AAL_s23
                                 - s_k10_PRU_s23  = 0
 k10_s24_balance_constraint:     b_k10_HSBC_s24 + b_k10_VOD_s24 + b_k10_BP_s24
                                 + b_k10_GSK_s24 + b_k10_AZN_s24
                                 + b_k10_RIO_s24 + b_k10_BG_s24
                                 + b_k10_TSCO_s24 + b_k10_AAL_s24
                                 + b_k10_PRU_s24 - s_k10_HSBC_s24
                                 - s_k10_VOD_s24 - s_k10_BP_s24 - s_k10_GSK_s24
                                 - s_k10_AZN_s24 - s_k10_RIO_s24 - s_k10_BG_s24
                                 - s_k10_TSCO_s24 - s_k10_AAL_s24
                                 - s_k10_PRU_s24  = 0
 k10_s25_balance_constraint:     b_k10_HSBC_s25 + b_k10_VOD_s25 + b_k10_BP_s25
                                 + b_k10_GSK_s25 + b_k10_AZN_s25
                                 + b_k10_RIO_s25 + b_k10_BG_s25
                                 + b_k10_TSCO_s25 + b_k10_AAL_s25
                                 + b_k10_PRU_s25 - s_k10_HSBC_s25
                                 - s_k10_VOD_s25 - s_k10_BP_s25 - s_k10_GSK_s25
                                 - s_k10_AZN_s25 - s_k10_RIO_s25 - s_k10_BG_s25
                                 - s_k10_TSCO_s25 - s_k10_AAL_s25
                                 - s_k10_PRU_s25  = 0
 k10_s26_balance_constraint:     b_k10_HSBC_s26 + b_k10_VOD_s26 + b_k10_BP_s26
                                 + b_k10_GSK_s26 + b_k10_AZN_s26
                                 + b_k10_RIO_s26 + b_k10_BG_s26
                                 + b_k10_TSCO_s26 + b_k10_AAL_s26
                                 + b_k10_PRU_s26 - s_k10_HSBC_s26
                                 - s_k10_VOD_s26 - s_k10_BP_s26 - s_k10_GSK_s26
                                 - s_k10_AZN_s26 - s_k10_RIO_s26 - s_k10_BG_s26
                                 - s_k10_TSCO_s26 - s_k10_AAL_s26
                                 - s_k10_PRU_s26  = 0
 k10_s27_balance_constraint:     b_k10_HSBC_s27 + b_k10_VOD_s27 + b_k10_BP_s27
                                 + b_k10_GSK_s27 + b_k10_AZN_s27
                                 + b_k10_RIO_s27 + b_k10_BG_s27
                                 + b_k10_TSCO_s27 + b_k10_AAL_s27
                                 + b_k10_PRU_s27 - s_k10_HSBC_s27
                                 - s_k10_VOD_s27 - s_k10_BP_s27 - s_k10_GSK_s27
                                 - s_k10_AZN_s27 - s_k10_RIO_s27 - s_k10_BG_s27
                                 - s_k10_TSCO_s27 - s_k10_AAL_s27
                                 - s_k10_PRU_s27  = 0
 k10_s28_balance_constraint:     b_k10_HSBC_s28 + b_k10_VOD_s28 + b_k10_BP_s28
                                 + b_k10_GSK_s28 + b_k10_AZN_s28
                                 + b_k10_RIO_s28 + b_k10_BG_s28
                                 + b_k10_TSCO_s28 + b_k10_AAL_s28
                                 + b_k10_PRU_s28 - s_k10_HSBC_s28
                                 - s_k10_VOD_s28 - s_k10_BP_s28 - s_k10_GSK_s28
                                 - s_k10_AZN_s28 - s_k10_RIO_s28 - s_k10_BG_s28
                                 - s_k10_TSCO_s28 - s_k10_AAL_s28
                                 - s_k10_PRU_s28  = 0
 k10_s29_balance_constraint:     b_k10_HSBC_s29 + b_k10_VOD_s29 + b_k10_BP_s29
                                 + b_k10_GSK_s29 + b_k10_AZN_s29
                                 + b_k10_RIO_s29 + b_k10_BG_s29
                                 + b_k10_TSCO_s29 + b_k10_AAL_s29
                                 + b_k10_PRU_s29 - s_k10_HSBC_s29
                                 - s_k10_VOD_s29 - s_k10_BP_s29 - s_k10_GSK_s29
                                 - s_k10_AZN_s29 - s_k10_RIO_s29 - s_k10_BG_s29
                                 - s_k10_TSCO_s29 - s_k10_AAL_s29
                                 - s_k10_PRU_s29  = 0
 k10_s30_balance_constraint:     b_k10_HSBC_s30 + b_k10_VOD_s30 + b_k10_BP_s30
                                 + b_k10_GSK_s30 + b_k10_AZN_s30
                                 + b_k10_RIO_s30 + b_k10_BG_s30
                                 + b_k10_TSCO_s30 + b_k10_AAL_s30
                                 + b_k10_PRU_s30 - s_k10_HSBC_s30
                                 - s_k10_VOD_s30 - s_k10_BP_s30 - s_k10_GSK_s30
                                 - s_k10_AZN_s30 - s_k10_RIO_s30 - s_k10_BG_s30
                                 - s_k10_TSCO_s30 - s_k10_AAL_s30
                                 - s_k10_PRU_s30  = 0
 k10_s31_balance_constraint:     b_k10_HSBC_s31 + b_k10_VOD_s31 + b_k10_BP_s31
                                 + b_k10_GSK_s31 + b_k10_AZN_s31
                                 + b_k10_RIO_s31 + b_k10_BG_s31
                                 + b_k10_TSCO_s31 + b_k10_AAL_s31
                                 + b_k10_PRU_s31 - s_k10_HSBC_s31
                                 - s_k10_VOD_s31 - s_k10_BP_s31 - s_k10_GSK_s31
                                 - s_k10_AZN_s31 - s_k10_RIO_s31 - s_k10_BG_s31
                                 - s_k10_TSCO_s31 - s_k10_AAL_s31
                                 - s_k10_PRU_s31  = 0
 k10_s32_balance_constraint:     b_k10_HSBC_s32 + b_k10_VOD_s32 + b_k10_BP_s32
                                 + b_k10_GSK_s32 + b_k10_AZN_s32
                                 + b_k10_RIO_s32 + b_k10_BG_s32
                                 + b_k10_TSCO_s32 + b_k10_AAL_s32
                                 + b_k10_PRU_s32 - s_k10_HSBC_s32
                                 - s_k10_VOD_s32 - s_k10_BP_s32 - s_k10_GSK_s32
                                 - s_k10_AZN_s32 - s_k10_RIO_s32 - s_k10_BG_s32
                                 - s_k10_TSCO_s32 - s_k10_AAL_s32
                                 - s_k10_PRU_s32  = 0
 k10_s33_balance_constraint:     b_k10_HSBC_s33 + b_k10_VOD_s33 + b_k10_BP_s33
                                 + b_k10_GSK_s33 + b_k10_AZN_s33
                                 + b_k10_RIO_s33 + b_k10_BG_s33
                                 + b_k10_TSCO_s33 + b_k10_AAL_s33
                                 + b_k10_PRU_s33 - s_k10_HSBC_s33
                                 - s_k10_VOD_s33 - s_k10_BP_s33 - s_k10_GSK_s33
                                 - s_k10_AZN_s33 - s_k10_RIO_s33 - s_k10_BG_s33
                                 - s_k10_TSCO_s33 - s_k10_AAL_s33
                                 - s_k10_PRU_s33  = 0
 k10_s34_balance_constraint:     b_k10_HSBC_s34 + b_k10_VOD_s34 + b_k10_BP_s34
                                 + b_k10_GSK_s34 + b_k10_AZN_s34
                                 + b_k10_RIO_s34 + b_k10_BG_s34
                                 + b_k10_TSCO_s34 + b_k10_AAL_s34
                                 + b_k10_PRU_s34 - s_k10_HSBC_s34
                                 - s_k10_VOD_s34 - s_k10_BP_s34 - s_k10_GSK_s34
                                 - s_k10_AZN_s34 - s_k10_RIO_s34 - s_k10_BG_s34
                                 - s_k10_TSCO_s34 - s_k10_AAL_s34
                                 - s_k10_PRU_s34  = 0
 k10_s35_balance_constraint:     b_k10_HSBC_s35 + b_k10_VOD_s35 + b_k10_BP_s35
                                 + b_k10_GSK_s35 + b_k10_AZN_s35
                                 + b_k10_RIO_s35 + b_k10_BG_s35
                                 + b_k10_TSCO_s35 + b_k10_AAL_s35
                                 + b_k10_PRU_s35 - s_k10_HSBC_s35
                                 - s_k10_VOD_s35 - s_k10_BP_s35 - s_k10_GSK_s35
                                 - s_k10_AZN_s35 - s_k10_RIO_s35 - s_k10_BG_s35
                                 - s_k10_TSCO_s35 - s_k10_AAL_s35
                                 - s_k10_PRU_s35  = 0
 k10_s36_balance_constraint:     b_k10_HSBC_s36 + b_k10_VOD_s36 + b_k10_BP_s36
                                 + b_k10_GSK_s36 + b_k10_AZN_s36
                                 + b_k10_RIO_s36 + b_k10_BG_s36
                                 + b_k10_TSCO_s36 + b_k10_AAL_s36
                                 + b_k10_PRU_s36 - s_k10_HSBC_s36
                                 - s_k10_VOD_s36 - s_k10_BP_s36 - s_k10_GSK_s36
                                 - s_k10_AZN_s36 - s_k10_RIO_s36 - s_k10_BG_s36
                                 - s_k10_TSCO_s36 - s_k10_AAL_s36
                                 - s_k10_PRU_s36  = 0
 k10_s37_balance_constraint:     b_k10_HSBC_s37 + b_k10_VOD_s37 + b_k10_BP_s37
                                 + b_k10_GSK_s37 + b_k10_AZN_s37
                                 + b_k10_RIO_s37 + b_k10_BG_s37
                                 + b_k10_TSCO_s37 + b_k10_AAL_s37
                                 + b_k10_PRU_s37 - s_k10_HSBC_s37
                                 - s_k10_VOD_s37 - s_k10_BP_s37 - s_k10_GSK_s37
                                 - s_k10_AZN_s37 - s_k10_RIO_s37 - s_k10_BG_s37
                                 - s_k10_TSCO_s37 - s_k10_AAL_s37
                                 - s_k10_PRU_s37  = 0
 k10_s38_balance_constraint:     b_k10_HSBC_s38 + b_k10_VOD_s38 + b_k10_BP_s38
                                 + b_k10_GSK_s38 + b_k10_AZN_s38
                                 + b_k10_RIO_s38 + b_k10_BG_s38
                                 + b_k10_TSCO_s38 + b_k10_AAL_s38
                                 + b_k10_PRU_s38 - s_k10_HSBC_s38
                                 - s_k10_VOD_s38 - s_k10_BP_s38 - s_k10_GSK_s38
                                 - s_k10_AZN_s38 - s_k10_RIO_s38 - s_k10_BG_s38
                                 - s_k10_TSCO_s38 - s_k10_AAL_s38
                                 - s_k10_PRU_s38  = 0
 k10_s39_balance_constraint:     b_k10_HSBC_s39 + b_k10_VOD_s39 + b_k10_BP_s39
                                 + b_k10_GSK_s39 + b_k10_AZN_s39
                                 + b_k10_RIO_s39 + b_k10_BG_s39
                                 + b_k10_TSCO_s39 + b_k10_AAL_s39
                                 + b_k10_PRU_s39 - s_k10_HSBC_s39
                                 - s_k10_VOD_s39 - s_k10_BP_s39 - s_k10_GSK_s39
                                 - s_k10_AZN_s39 - s_k10_RIO_s39 - s_k10_BG_s39
                                 - s_k10_TSCO_s39 - s_k10_AAL_s39
                                 - s_k10_PRU_s39  = 0
 k10_s40_balance_constraint:     b_k10_HSBC_s40 + b_k10_VOD_s40 + b_k10_BP_s40
                                 + b_k10_GSK_s40 + b_k10_AZN_s40
                                 + b_k10_RIO_s40 + b_k10_BG_s40
                                 + b_k10_TSCO_s40 + b_k10_AAL_s40
                                 + b_k10_PRU_s40 - s_k10_HSBC_s40
                                 - s_k10_VOD_s40 - s_k10_BP_s40 - s_k10_GSK_s40
                                 - s_k10_AZN_s40 - s_k10_RIO_s40 - s_k10_BG_s40
                                 - s_k10_TSCO_s40 - s_k10_AAL_s40
                                 - s_k10_PRU_s40  = 0
 k10_s41_balance_constraint:     b_k10_HSBC_s41 + b_k10_VOD_s41 + b_k10_BP_s41
                                 + b_k10_GSK_s41 + b_k10_AZN_s41
                                 + b_k10_RIO_s41 + b_k10_BG_s41
                                 + b_k10_TSCO_s41 + b_k10_AAL_s41
                                 + b_k10_PRU_s41 - s_k10_HSBC_s41
                                 - s_k10_VOD_s41 - s_k10_BP_s41 - s_k10_GSK_s41
                                 - s_k10_AZN_s41 - s_k10_RIO_s41 - s_k10_BG_s41
                                 - s_k10_TSCO_s41 - s_k10_AAL_s41
                                 - s_k10_PRU_s41  = 0
 k10_s42_balance_constraint:     b_k10_HSBC_s42 + b_k10_VOD_s42 + b_k10_BP_s42
                                 + b_k10_GSK_s42 + b_k10_AZN_s42
                                 + b_k10_RIO_s42 + b_k10_BG_s42
                                 + b_k10_TSCO_s42 + b_k10_AAL_s42
                                 + b_k10_PRU_s42 - s_k10_HSBC_s42
                                 - s_k10_VOD_s42 - s_k10_BP_s42 - s_k10_GSK_s42
                                 - s_k10_AZN_s42 - s_k10_RIO_s42 - s_k10_BG_s42
                                 - s_k10_TSCO_s42 - s_k10_AAL_s42
                                 - s_k10_PRU_s42  = 0
 k10_s43_balance_constraint:     b_k10_HSBC_s43 + b_k10_VOD_s43 + b_k10_BP_s43
                                 + b_k10_GSK_s43 + b_k10_AZN_s43
                                 + b_k10_RIO_s43 + b_k10_BG_s43
                                 + b_k10_TSCO_s43 + b_k10_AAL_s43
                                 + b_k10_PRU_s43 - s_k10_HSBC_s43
                                 - s_k10_VOD_s43 - s_k10_BP_s43 - s_k10_GSK_s43
                                 - s_k10_AZN_s43 - s_k10_RIO_s43 - s_k10_BG_s43
                                 - s_k10_TSCO_s43 - s_k10_AAL_s43
                                 - s_k10_PRU_s43  = 0
 k10_s44_balance_constraint:     b_k10_HSBC_s44 + b_k10_VOD_s44 + b_k10_BP_s44
                                 + b_k10_GSK_s44 + b_k10_AZN_s44
                                 + b_k10_RIO_s44 + b_k10_BG_s44
                                 + b_k10_TSCO_s44 + b_k10_AAL_s44
                                 + b_k10_PRU_s44 - s_k10_HSBC_s44
                                 - s_k10_VOD_s44 - s_k10_BP_s44 - s_k10_GSK_s44
                                 - s_k10_AZN_s44 - s_k10_RIO_s44 - s_k10_BG_s44
                                 - s_k10_TSCO_s44 - s_k10_AAL_s44
                                 - s_k10_PRU_s44  = 0
 k10_s45_balance_constraint:     b_k10_HSBC_s45 + b_k10_VOD_s45 + b_k10_BP_s45
                                 + b_k10_GSK_s45 + b_k10_AZN_s45
                                 + b_k10_RIO_s45 + b_k10_BG_s45
                                 + b_k10_TSCO_s45 + b_k10_AAL_s45
                                 + b_k10_PRU_s45 - s_k10_HSBC_s45
                                 - s_k10_VOD_s45 - s_k10_BP_s45 - s_k10_GSK_s45
                                 - s_k10_AZN_s45 - s_k10_RIO_s45 - s_k10_BG_s45
                                 - s_k10_TSCO_s45 - s_k10_AAL_s45
                                 - s_k10_PRU_s45  = 0
 k10_s46_balance_constraint:     b_k10_HSBC_s46 + b_k10_VOD_s46 + b_k10_BP_s46
                                 + b_k10_GSK_s46 + b_k10_AZN_s46
                                 + b_k10_RIO_s46 + b_k10_BG_s46
                                 + b_k10_TSCO_s46 + b_k10_AAL_s46
                                 + b_k10_PRU_s46 - s_k10_HSBC_s46
                                 - s_k10_VOD_s46 - s_k10_BP_s46 - s_k10_GSK_s46
                                 - s_k10_AZN_s46 - s_k10_RIO_s46 - s_k10_BG_s46
                                 - s_k10_TSCO_s46 - s_k10_AAL_s46
                                 - s_k10_PRU_s46  = 0
 k10_s47_balance_constraint:     b_k10_HSBC_s47 + b_k10_VOD_s47 + b_k10_BP_s47
                                 + b_k10_GSK_s47 + b_k10_AZN_s47
                                 + b_k10_RIO_s47 + b_k10_BG_s47
                                 + b_k10_TSCO_s47 + b_k10_AAL_s47
                                 + b_k10_PRU_s47 - s_k10_HSBC_s47
                                 - s_k10_VOD_s47 - s_k10_BP_s47 - s_k10_GSK_s47
                                 - s_k10_AZN_s47 - s_k10_RIO_s47 - s_k10_BG_s47
                                 - s_k10_TSCO_s47 - s_k10_AAL_s47
                                 - s_k10_PRU_s47  = 0
 k10_s48_balance_constraint:     b_k10_HSBC_s48 + b_k10_VOD_s48 + b_k10_BP_s48
                                 + b_k10_GSK_s48 + b_k10_AZN_s48
                                 + b_k10_RIO_s48 + b_k10_BG_s48
                                 + b_k10_TSCO_s48 + b_k10_AAL_s48
                                 + b_k10_PRU_s48 - s_k10_HSBC_s48
                                 - s_k10_VOD_s48 - s_k10_BP_s48 - s_k10_GSK_s48
                                 - s_k10_AZN_s48 - s_k10_RIO_s48 - s_k10_BG_s48
                                 - s_k10_TSCO_s48 - s_k10_AAL_s48
                                 - s_k10_PRU_s48  = 0
 k10_s49_balance_constraint:     b_k10_HSBC_s49 + b_k10_VOD_s49 + b_k10_BP_s49
                                 + b_k10_GSK_s49 + b_k10_AZN_s49
                                 + b_k10_RIO_s49 + b_k10_BG_s49
                                 + b_k10_TSCO_s49 + b_k10_AAL_s49
                                 + b_k10_PRU_s49 - s_k10_HSBC_s49
                                 - s_k10_VOD_s49 - s_k10_BP_s49 - s_k10_GSK_s49
                                 - s_k10_AZN_s49 - s_k10_RIO_s49 - s_k10_BG_s49
                                 - s_k10_TSCO_s49 - s_k10_AAL_s49
                                 - s_k10_PRU_s49  = 0
 k10_s50_balance_constraint:     b_k10_HSBC_s50 + b_k10_VOD_s50 + b_k10_BP_s50
                                 + b_k10_GSK_s50 + b_k10_AZN_s50
                                 + b_k10_RIO_s50 + b_k10_BG_s50
                                 + b_k10_TSCO_s50 + b_k10_AAL_s50
                                 + b_k10_PRU_s50 - s_k10_HSBC_s50
                                 - s_k10_VOD_s50 - s_k10_BP_s50 - s_k10_GSK_s50
                                 - s_k10_AZN_s50 - s_k10_RIO_s50 - s_k10_BG_s50
                                 - s_k10_TSCO_s50 - s_k10_AAL_s50
                                 - s_k10_PRU_s50  = 0
 k10_s51_balance_constraint:     b_k10_HSBC_s51 + b_k10_VOD_s51 + b_k10_BP_s51
                                 + b_k10_GSK_s51 + b_k10_AZN_s51
                                 + b_k10_RIO_s51 + b_k10_BG_s51
                                 + b_k10_TSCO_s51 + b_k10_AAL_s51
                                 + b_k10_PRU_s51 - s_k10_HSBC_s51
                                 - s_k10_VOD_s51 - s_k10_BP_s51 - s_k10_GSK_s51
                                 - s_k10_AZN_s51 - s_k10_RIO_s51 - s_k10_BG_s51
                                 - s_k10_TSCO_s51 - s_k10_AAL_s51
                                 - s_k10_PRU_s51  = 0
 k10_s52_balance_constraint:     b_k10_HSBC_s52 + b_k10_VOD_s52 + b_k10_BP_s52
                                 + b_k10_GSK_s52 + b_k10_AZN_s52
                                 + b_k10_RIO_s52 + b_k10_BG_s52
                                 + b_k10_TSCO_s52 + b_k10_AAL_s52
                                 + b_k10_PRU_s52 - s_k10_HSBC_s52
                                 - s_k10_VOD_s52 - s_k10_BP_s52 - s_k10_GSK_s52
                                 - s_k10_AZN_s52 - s_k10_RIO_s52 - s_k10_BG_s52
                                 - s_k10_TSCO_s52 - s_k10_AAL_s52
                                 - s_k10_PRU_s52  = 0
 k10_s53_balance_constraint:     b_k10_HSBC_s53 + b_k10_VOD_s53 + b_k10_BP_s53
                                 + b_k10_GSK_s53 + b_k10_AZN_s53
                                 + b_k10_RIO_s53 + b_k10_BG_s53
                                 + b_k10_TSCO_s53 + b_k10_AAL_s53
                                 + b_k10_PRU_s53 - s_k10_HSBC_s53
                                 - s_k10_VOD_s53 - s_k10_BP_s53 - s_k10_GSK_s53
                                 - s_k10_AZN_s53 - s_k10_RIO_s53 - s_k10_BG_s53
                                 - s_k10_TSCO_s53 - s_k10_AAL_s53
                                 - s_k10_PRU_s53  = 0
 k10_s54_balance_constraint:     b_k10_HSBC_s54 + b_k10_VOD_s54 + b_k10_BP_s54
                                 + b_k10_GSK_s54 + b_k10_AZN_s54
                                 + b_k10_RIO_s54 + b_k10_BG_s54
                                 + b_k10_TSCO_s54 + b_k10_AAL_s54
                                 + b_k10_PRU_s54 - s_k10_HSBC_s54
                                 - s_k10_VOD_s54 - s_k10_BP_s54 - s_k10_GSK_s54
                                 - s_k10_AZN_s54 - s_k10_RIO_s54 - s_k10_BG_s54
                                 - s_k10_TSCO_s54 - s_k10_AAL_s54
                                 - s_k10_PRU_s54  = 0
 k10_s55_balance_constraint:     b_k10_HSBC_s55 + b_k10_VOD_s55 + b_k10_BP_s55
                                 + b_k10_GSK_s55 + b_k10_AZN_s55
                                 + b_k10_RIO_s55 + b_k10_BG_s55
                                 + b_k10_TSCO_s55 + b_k10_AAL_s55
                                 + b_k10_PRU_s55 - s_k10_HSBC_s55
                                 - s_k10_VOD_s55 - s_k10_BP_s55 - s_k10_GSK_s55
                                 - s_k10_AZN_s55 - s_k10_RIO_s55 - s_k10_BG_s55
                                 - s_k10_TSCO_s55 - s_k10_AAL_s55
                                 - s_k10_PRU_s55  = 0
 k10_s56_balance_constraint:     b_k10_HSBC_s56 + b_k10_VOD_s56 + b_k10_BP_s56
                                 + b_k10_GSK_s56 + b_k10_AZN_s56
                                 + b_k10_RIO_s56 + b_k10_BG_s56
                                 + b_k10_TSCO_s56 + b_k10_AAL_s56
                                 + b_k10_PRU_s56 - s_k10_HSBC_s56
                                 - s_k10_VOD_s56 - s_k10_BP_s56 - s_k10_GSK_s56
                                 - s_k10_AZN_s56 - s_k10_RIO_s56 - s_k10_BG_s56
                                 - s_k10_TSCO_s56 - s_k10_AAL_s56
                                 - s_k10_PRU_s56  = 0
 k10_s57_balance_constraint:     b_k10_HSBC_s57 + b_k10_VOD_s57 + b_k10_BP_s57
                                 + b_k10_GSK_s57 + b_k10_AZN_s57
                                 + b_k10_RIO_s57 + b_k10_BG_s57
                                 + b_k10_TSCO_s57 + b_k10_AAL_s57
                                 + b_k10_PRU_s57 - s_k10_HSBC_s57
                                 - s_k10_VOD_s57 - s_k10_BP_s57 - s_k10_GSK_s57
                                 - s_k10_AZN_s57 - s_k10_RIO_s57 - s_k10_BG_s57
                                 - s_k10_TSCO_s57 - s_k10_AAL_s57
                                 - s_k10_PRU_s57  = 0
 k10_s58_balance_constraint:     b_k10_HSBC_s58 + b_k10_VOD_s58 + b_k10_BP_s58
                                 + b_k10_GSK_s58 + b_k10_AZN_s58
                                 + b_k10_RIO_s58 + b_k10_BG_s58
                                 + b_k10_TSCO_s58 + b_k10_AAL_s58
                                 + b_k10_PRU_s58 - s_k10_HSBC_s58
                                 - s_k10_VOD_s58 - s_k10_BP_s58 - s_k10_GSK_s58
                                 - s_k10_AZN_s58 - s_k10_RIO_s58 - s_k10_BG_s58
                                 - s_k10_TSCO_s58 - s_k10_AAL_s58
                                 - s_k10_PRU_s58  = 0
 k10_s59_balance_constraint:     b_k10_HSBC_s59 + b_k10_VOD_s59 + b_k10_BP_s59
                                 + b_k10_GSK_s59 + b_k10_AZN_s59
                                 + b_k10_RIO_s59 + b_k10_BG_s59
                                 + b_k10_TSCO_s59 + b_k10_AAL_s59
                                 + b_k10_PRU_s59 - s_k10_HSBC_s59
                                 - s_k10_VOD_s59 - s_k10_BP_s59 - s_k10_GSK_s59
                                 - s_k10_AZN_s59 - s_k10_RIO_s59 - s_k10_BG_s59
                                 - s_k10_TSCO_s59 - s_k10_AAL_s59
                                 - s_k10_PRU_s59  = 0
 k10_s60_balance_constraint:     b_k10_HSBC_s60 + b_k10_VOD_s60 + b_k10_BP_s60
                                 + b_k10_GSK_s60 + b_k10_AZN_s60
                                 + b_k10_RIO_s60 + b_k10_BG_s60
                                 + b_k10_TSCO_s60 + b_k10_AAL_s60
                                 + b_k10_PRU_s60 - s_k10_HSBC_s60
                                 - s_k10_VOD_s60 - s_k10_BP_s60 - s_k10_GSK_s60
                                 - s_k10_AZN_s60 - s_k10_RIO_s60 - s_k10_BG_s60
                                 - s_k10_TSCO_s60 - s_k10_AAL_s60
                                 - s_k10_PRU_s60  = 0
 k10_s61_balance_constraint:     b_k10_HSBC_s61 + b_k10_VOD_s61 + b_k10_BP_s61
                                 + b_k10_GSK_s61 + b_k10_AZN_s61
                                 + b_k10_RIO_s61 + b_k10_BG_s61
                                 + b_k10_TSCO_s61 + b_k10_AAL_s61
                                 + b_k10_PRU_s61 - s_k10_HSBC_s61
                                 - s_k10_VOD_s61 - s_k10_BP_s61 - s_k10_GSK_s61
                                 - s_k10_AZN_s61 - s_k10_RIO_s61 - s_k10_BG_s61
                                 - s_k10_TSCO_s61 - s_k10_AAL_s61
                                 - s_k10_PRU_s61  = 0
 k10_s62_balance_constraint:     b_k10_HSBC_s62 + b_k10_VOD_s62 + b_k10_BP_s62
                                 + b_k10_GSK_s62 + b_k10_AZN_s62
                                 + b_k10_RIO_s62 + b_k10_BG_s62
                                 + b_k10_TSCO_s62 + b_k10_AAL_s62
                                 + b_k10_PRU_s62 - s_k10_HSBC_s62
                                 - s_k10_VOD_s62 - s_k10_BP_s62 - s_k10_GSK_s62
                                 - s_k10_AZN_s62 - s_k10_RIO_s62 - s_k10_BG_s62
                                 - s_k10_TSCO_s62 - s_k10_AAL_s62
                                 - s_k10_PRU_s62  = 0
 k10_s63_balance_constraint:     b_k10_HSBC_s63 + b_k10_VOD_s63 + b_k10_BP_s63
                                 + b_k10_GSK_s63 + b_k10_AZN_s63
                                 + b_k10_RIO_s63 + b_k10_BG_s63
                                 + b_k10_TSCO_s63 + b_k10_AAL_s63
                                 + b_k10_PRU_s63 - s_k10_HSBC_s63
                                 - s_k10_VOD_s63 - s_k10_BP_s63 - s_k10_GSK_s63
                                 - s_k10_AZN_s63 - s_k10_RIO_s63 - s_k10_BG_s63
                                 - s_k10_TSCO_s63 - s_k10_AAL_s63
                                 - s_k10_PRU_s63  = 0
 k10_s64_balance_constraint:     b_k10_HSBC_s64 + b_k10_VOD_s64 + b_k10_BP_s64
                                 + b_k10_GSK_s64 + b_k10_AZN_s64
                                 + b_k10_RIO_s64 + b_k10_BG_s64
                                 + b_k10_TSCO_s64 + b_k10_AAL_s64
                                 + b_k10_PRU_s64 - s_k10_HSBC_s64
                                 - s_k10_VOD_s64 - s_k10_BP_s64 - s_k10_GSK_s64
                                 - s_k10_AZN_s64 - s_k10_RIO_s64 - s_k10_BG_s64
                                 - s_k10_TSCO_s64 - s_k10_AAL_s64
                                 - s_k10_PRU_s64  = 0
 k10_s65_balance_constraint:     b_k10_HSBC_s65 + b_k10_VOD_s65 + b_k10_BP_s65
                                 + b_k10_GSK_s65 + b_k10_AZN_s65
                                 + b_k10_RIO_s65 + b_k10_BG_s65
                                 + b_k10_TSCO_s65 + b_k10_AAL_s65
                                 + b_k10_PRU_s65 - s_k10_HSBC_s65
                                 - s_k10_VOD_s65 - s_k10_BP_s65 - s_k10_GSK_s65
                                 - s_k10_AZN_s65 - s_k10_RIO_s65 - s_k10_BG_s65
                                 - s_k10_TSCO_s65 - s_k10_AAL_s65
                                 - s_k10_PRU_s65  = 0
 k10_s66_balance_constraint:     b_k10_HSBC_s66 + b_k10_VOD_s66 + b_k10_BP_s66
                                 + b_k10_GSK_s66 + b_k10_AZN_s66
                                 + b_k10_RIO_s66 + b_k10_BG_s66
                                 + b_k10_TSCO_s66 + b_k10_AAL_s66
                                 + b_k10_PRU_s66 - s_k10_HSBC_s66
                                 - s_k10_VOD_s66 - s_k10_BP_s66 - s_k10_GSK_s66
                                 - s_k10_AZN_s66 - s_k10_RIO_s66 - s_k10_BG_s66
                                 - s_k10_TSCO_s66 - s_k10_AAL_s66
                                 - s_k10_PRU_s66  = 0
 k10_s67_balance_constraint:     b_k10_HSBC_s67 + b_k10_VOD_s67 + b_k10_BP_s67
                                 + b_k10_GSK_s67 + b_k10_AZN_s67
                                 + b_k10_RIO_s67 + b_k10_BG_s67
                                 + b_k10_TSCO_s67 + b_k10_AAL_s67
                                 + b_k10_PRU_s67 - s_k10_HSBC_s67
                                 - s_k10_VOD_s67 - s_k10_BP_s67 - s_k10_GSK_s67
                                 - s_k10_AZN_s67 - s_k10_RIO_s67 - s_k10_BG_s67
                                 - s_k10_TSCO_s67 - s_k10_AAL_s67
                                 - s_k10_PRU_s67  = 0
 k10_s68_balance_constraint:     b_k10_HSBC_s68 + b_k10_VOD_s68 + b_k10_BP_s68
                                 + b_k10_GSK_s68 + b_k10_AZN_s68
                                 + b_k10_RIO_s68 + b_k10_BG_s68
                                 + b_k10_TSCO_s68 + b_k10_AAL_s68
                                 + b_k10_PRU_s68 - s_k10_HSBC_s68
                                 - s_k10_VOD_s68 - s_k10_BP_s68 - s_k10_GSK_s68
                                 - s_k10_AZN_s68 - s_k10_RIO_s68 - s_k10_BG_s68
                                 - s_k10_TSCO_s68 - s_k10_AAL_s68
                                 - s_k10_PRU_s68  = 0
 k10_s69_balance_constraint:     b_k10_HSBC_s69 + b_k10_VOD_s69 + b_k10_BP_s69
                                 + b_k10_GSK_s69 + b_k10_AZN_s69
                                 + b_k10_RIO_s69 + b_k10_BG_s69
                                 + b_k10_TSCO_s69 + b_k10_AAL_s69
                                 + b_k10_PRU_s69 - s_k10_HSBC_s69
                                 - s_k10_VOD_s69 - s_k10_BP_s69 - s_k10_GSK_s69
                                 - s_k10_AZN_s69 - s_k10_RIO_s69 - s_k10_BG_s69
                                 - s_k10_TSCO_s69 - s_k10_AAL_s69
                                 - s_k10_PRU_s69  = 0
 k10_s70_balance_constraint:     b_k10_HSBC_s70 + b_k10_VOD_s70 + b_k10_BP_s70
                                 + b_k10_GSK_s70 + b_k10_AZN_s70
                                 + b_k10_RIO_s70 + b_k10_BG_s70
                                 + b_k10_TSCO_s70 + b_k10_AAL_s70
                                 + b_k10_PRU_s70 - s_k10_HSBC_s70
                                 - s_k10_VOD_s70 - s_k10_BP_s70 - s_k10_GSK_s70
                                 - s_k10_AZN_s70 - s_k10_RIO_s70 - s_k10_BG_s70
                                 - s_k10_TSCO_s70 - s_k10_AAL_s70
                                 - s_k10_PRU_s70  = 0
 k10_s71_balance_constraint:     b_k10_HSBC_s71 + b_k10_VOD_s71 + b_k10_BP_s71
                                 + b_k10_GSK_s71 + b_k10_AZN_s71
                                 + b_k10_RIO_s71 + b_k10_BG_s71
                                 + b_k10_TSCO_s71 + b_k10_AAL_s71
                                 + b_k10_PRU_s71 - s_k10_HSBC_s71
                                 - s_k10_VOD_s71 - s_k10_BP_s71 - s_k10_GSK_s71
                                 - s_k10_AZN_s71 - s_k10_RIO_s71 - s_k10_BG_s71
                                 - s_k10_TSCO_s71 - s_k10_AAL_s71
                                 - s_k10_PRU_s71  = 0
 k10_s72_balance_constraint:     b_k10_HSBC_s72 + b_k10_VOD_s72 + b_k10_BP_s72
                                 + b_k10_GSK_s72 + b_k10_AZN_s72
                                 + b_k10_RIO_s72 + b_k10_BG_s72
                                 + b_k10_TSCO_s72 + b_k10_AAL_s72
                                 + b_k10_PRU_s72 - s_k10_HSBC_s72
                                 - s_k10_VOD_s72 - s_k10_BP_s72 - s_k10_GSK_s72
                                 - s_k10_AZN_s72 - s_k10_RIO_s72 - s_k10_BG_s72
                                 - s_k10_TSCO_s72 - s_k10_AAL_s72
                                 - s_k10_PRU_s72  = 0
 k10_s73_balance_constraint:     b_k10_HSBC_s73 + b_k10_VOD_s73 + b_k10_BP_s73
                                 + b_k10_GSK_s73 + b_k10_AZN_s73
                                 + b_k10_RIO_s73 + b_k10_BG_s73
                                 + b_k10_TSCO_s73 + b_k10_AAL_s73
                                 + b_k10_PRU_s73 - s_k10_HSBC_s73
                                 - s_k10_VOD_s73 - s_k10_BP_s73 - s_k10_GSK_s73
                                 - s_k10_AZN_s73 - s_k10_RIO_s73 - s_k10_BG_s73
                                 - s_k10_TSCO_s73 - s_k10_AAL_s73
                                 - s_k10_PRU_s73  = 0
 k10_s74_balance_constraint:     b_k10_HSBC_s74 + b_k10_VOD_s74 + b_k10_BP_s74
                                 + b_k10_GSK_s74 + b_k10_AZN_s74
                                 + b_k10_RIO_s74 + b_k10_BG_s74
                                 + b_k10_TSCO_s74 + b_k10_AAL_s74
                                 + b_k10_PRU_s74 - s_k10_HSBC_s74
                                 - s_k10_VOD_s74 - s_k10_BP_s74 - s_k10_GSK_s74
                                 - s_k10_AZN_s74 - s_k10_RIO_s74 - s_k10_BG_s74
                                 - s_k10_TSCO_s74 - s_k10_AAL_s74
                                 - s_k10_PRU_s74  = 0
 k10_s75_balance_constraint:     b_k10_HSBC_s75 + b_k10_VOD_s75 + b_k10_BP_s75
                                 + b_k10_GSK_s75 + b_k10_AZN_s75
                                 + b_k10_RIO_s75 + b_k10_BG_s75
                                 + b_k10_TSCO_s75 + b_k10_AAL_s75
                                 + b_k10_PRU_s75 - s_k10_HSBC_s75
                                 - s_k10_VOD_s75 - s_k10_BP_s75 - s_k10_GSK_s75
                                 - s_k10_AZN_s75 - s_k10_RIO_s75 - s_k10_BG_s75
                                 - s_k10_TSCO_s75 - s_k10_AAL_s75
                                 - s_k10_PRU_s75  = 0
 k10_s76_balance_constraint:     b_k10_HSBC_s76 + b_k10_VOD_s76 + b_k10_BP_s76
                                 + b_k10_GSK_s76 + b_k10_AZN_s76
                                 + b_k10_RIO_s76 + b_k10_BG_s76
                                 + b_k10_TSCO_s76 + b_k10_AAL_s76
                                 + b_k10_PRU_s76 - s_k10_HSBC_s76
                                 - s_k10_VOD_s76 - s_k10_BP_s76 - s_k10_GSK_s76
                                 - s_k10_AZN_s76 - s_k10_RIO_s76 - s_k10_BG_s76
                                 - s_k10_TSCO_s76 - s_k10_AAL_s76
                                 - s_k10_PRU_s76  = 0
 k10_s77_balance_constraint:     b_k10_HSBC_s77 + b_k10_VOD_s77 + b_k10_BP_s77
                                 + b_k10_GSK_s77 + b_k10_AZN_s77
                                 + b_k10_RIO_s77 + b_k10_BG_s77
                                 + b_k10_TSCO_s77 + b_k10_AAL_s77
                                 + b_k10_PRU_s77 - s_k10_HSBC_s77
                                 - s_k10_VOD_s77 - s_k10_BP_s77 - s_k10_GSK_s77
                                 - s_k10_AZN_s77 - s_k10_RIO_s77 - s_k10_BG_s77
                                 - s_k10_TSCO_s77 - s_k10_AAL_s77
                                 - s_k10_PRU_s77  = 0
 k10_s78_balance_constraint:     b_k10_HSBC_s78 + b_k10_VOD_s78 + b_k10_BP_s78
                                 + b_k10_GSK_s78 + b_k10_AZN_s78
                                 + b_k10_RIO_s78 + b_k10_BG_s78
                                 + b_k10_TSCO_s78 + b_k10_AAL_s78
                                 + b_k10_PRU_s78 - s_k10_HSBC_s78
                                 - s_k10_VOD_s78 - s_k10_BP_s78 - s_k10_GSK_s78
                                 - s_k10_AZN_s78 - s_k10_RIO_s78 - s_k10_BG_s78
                                 - s_k10_TSCO_s78 - s_k10_AAL_s78
                                 - s_k10_PRU_s78  = 0
 k10_s79_balance_constraint:     b_k10_HSBC_s79 + b_k10_VOD_s79 + b_k10_BP_s79
                                 + b_k10_GSK_s79 + b_k10_AZN_s79
                                 + b_k10_RIO_s79 + b_k10_BG_s79
                                 + b_k10_TSCO_s79 + b_k10_AAL_s79
                                 + b_k10_PRU_s79 - s_k10_HSBC_s79
                                 - s_k10_VOD_s79 - s_k10_BP_s79 - s_k10_GSK_s79
                                 - s_k10_AZN_s79 - s_k10_RIO_s79 - s_k10_BG_s79
                                 - s_k10_TSCO_s79 - s_k10_AAL_s79
                                 - s_k10_PRU_s79  = 0
 k10_s80_balance_constraint:     b_k10_HSBC_s80 + b_k10_VOD_s80 + b_k10_BP_s80
                                 + b_k10_GSK_s80 + b_k10_AZN_s80
                                 + b_k10_RIO_s80 + b_k10_BG_s80
                                 + b_k10_TSCO_s80 + b_k10_AAL_s80
                                 + b_k10_PRU_s80 - s_k10_HSBC_s80
                                 - s_k10_VOD_s80 - s_k10_BP_s80 - s_k10_GSK_s80
                                 - s_k10_AZN_s80 - s_k10_RIO_s80 - s_k10_BG_s80
                                 - s_k10_TSCO_s80 - s_k10_AAL_s80
                                 - s_k10_PRU_s80  = 0
 k10_s81_balance_constraint:     b_k10_HSBC_s81 + b_k10_VOD_s81 + b_k10_BP_s81
                                 + b_k10_GSK_s81 + b_k10_AZN_s81
                                 + b_k10_RIO_s81 + b_k10_BG_s81
                                 + b_k10_TSCO_s81 + b_k10_AAL_s81
                                 + b_k10_PRU_s81 - s_k10_HSBC_s81
                                 - s_k10_VOD_s81 - s_k10_BP_s81 - s_k10_GSK_s81
                                 - s_k10_AZN_s81 - s_k10_RIO_s81 - s_k10_BG_s81
                                 - s_k10_TSCO_s81 - s_k10_AAL_s81
                                 - s_k10_PRU_s81  = 0
 k10_s82_balance_constraint:     b_k10_HSBC_s82 + b_k10_VOD_s82 + b_k10_BP_s82
                                 + b_k10_GSK_s82 + b_k10_AZN_s82
                                 + b_k10_RIO_s82 + b_k10_BG_s82
                                 + b_k10_TSCO_s82 + b_k10_AAL_s82
                                 + b_k10_PRU_s82 - s_k10_HSBC_s82
                                 - s_k10_VOD_s82 - s_k10_BP_s82 - s_k10_GSK_s82
                                 - s_k10_AZN_s82 - s_k10_RIO_s82 - s_k10_BG_s82
                                 - s_k10_TSCO_s82 - s_k10_AAL_s82
                                 - s_k10_PRU_s82  = 0
 k10_s83_balance_constraint:     b_k10_HSBC_s83 + b_k10_VOD_s83 + b_k10_BP_s83
                                 + b_k10_GSK_s83 + b_k10_AZN_s83
                                 + b_k10_RIO_s83 + b_k10_BG_s83
                                 + b_k10_TSCO_s83 + b_k10_AAL_s83
                                 + b_k10_PRU_s83 - s_k10_HSBC_s83
                                 - s_k10_VOD_s83 - s_k10_BP_s83 - s_k10_GSK_s83
                                 - s_k10_AZN_s83 - s_k10_RIO_s83 - s_k10_BG_s83
                                 - s_k10_TSCO_s83 - s_k10_AAL_s83
                                 - s_k10_PRU_s83  = 0
 k10_s84_balance_constraint:     b_k10_HSBC_s84 + b_k10_VOD_s84 + b_k10_BP_s84
                                 + b_k10_GSK_s84 + b_k10_AZN_s84
                                 + b_k10_RIO_s84 + b_k10_BG_s84
                                 + b_k10_TSCO_s84 + b_k10_AAL_s84
                                 + b_k10_PRU_s84 - s_k10_HSBC_s84
                                 - s_k10_VOD_s84 - s_k10_BP_s84 - s_k10_GSK_s84
                                 - s_k10_AZN_s84 - s_k10_RIO_s84 - s_k10_BG_s84
                                 - s_k10_TSCO_s84 - s_k10_AAL_s84
                                 - s_k10_PRU_s84  = 0
 k10_s85_balance_constraint:     b_k10_HSBC_s85 + b_k10_VOD_s85 + b_k10_BP_s85
                                 + b_k10_GSK_s85 + b_k10_AZN_s85
                                 + b_k10_RIO_s85 + b_k10_BG_s85
                                 + b_k10_TSCO_s85 + b_k10_AAL_s85
                                 + b_k10_PRU_s85 - s_k10_HSBC_s85
                                 - s_k10_VOD_s85 - s_k10_BP_s85 - s_k10_GSK_s85
                                 - s_k10_AZN_s85 - s_k10_RIO_s85 - s_k10_BG_s85
                                 - s_k10_TSCO_s85 - s_k10_AAL_s85
                                 - s_k10_PRU_s85  = 0
 k10_s86_balance_constraint:     b_k10_HSBC_s86 + b_k10_VOD_s86 + b_k10_BP_s86
                                 + b_k10_GSK_s86 + b_k10_AZN_s86
                                 + b_k10_RIO_s86 + b_k10_BG_s86
                                 + b_k10_TSCO_s86 + b_k10_AAL_s86
                                 + b_k10_PRU_s86 - s_k10_HSBC_s86
                                 - s_k10_VOD_s86 - s_k10_BP_s86 - s_k10_GSK_s86
                                 - s_k10_AZN_s86 - s_k10_RIO_s86 - s_k10_BG_s86
                                 - s_k10_TSCO_s86 - s_k10_AAL_s86
                                 - s_k10_PRU_s86  = 0
 k10_s87_balance_constraint:     b_k10_HSBC_s87 + b_k10_VOD_s87 + b_k10_BP_s87
                                 + b_k10_GSK_s87 + b_k10_AZN_s87
                                 + b_k10_RIO_s87 + b_k10_BG_s87
                                 + b_k10_TSCO_s87 + b_k10_AAL_s87
                                 + b_k10_PRU_s87 - s_k10_HSBC_s87
                                 - s_k10_VOD_s87 - s_k10_BP_s87 - s_k10_GSK_s87
                                 - s_k10_AZN_s87 - s_k10_RIO_s87 - s_k10_BG_s87
                                 - s_k10_TSCO_s87 - s_k10_AAL_s87
                                 - s_k10_PRU_s87  = 0
 k10_s88_balance_constraint:     b_k10_HSBC_s88 + b_k10_VOD_s88 + b_k10_BP_s88
                                 + b_k10_GSK_s88 + b_k10_AZN_s88
                                 + b_k10_RIO_s88 + b_k10_BG_s88
                                 + b_k10_TSCO_s88 + b_k10_AAL_s88
                                 + b_k10_PRU_s88 - s_k10_HSBC_s88
                                 - s_k10_VOD_s88 - s_k10_BP_s88 - s_k10_GSK_s88
                                 - s_k10_AZN_s88 - s_k10_RIO_s88 - s_k10_BG_s88
                                 - s_k10_TSCO_s88 - s_k10_AAL_s88
                                 - s_k10_PRU_s88  = 0
 k10_s89_balance_constraint:     b_k10_HSBC_s89 + b_k10_VOD_s89 + b_k10_BP_s89
                                 + b_k10_GSK_s89 + b_k10_AZN_s89
                                 + b_k10_RIO_s89 + b_k10_BG_s89
                                 + b_k10_TSCO_s89 + b_k10_AAL_s89
                                 + b_k10_PRU_s89 - s_k10_HSBC_s89
                                 - s_k10_VOD_s89 - s_k10_BP_s89 - s_k10_GSK_s89
                                 - s_k10_AZN_s89 - s_k10_RIO_s89 - s_k10_BG_s89
                                 - s_k10_TSCO_s89 - s_k10_AAL_s89
                                 - s_k10_PRU_s89  = 0
 k10_s90_balance_constraint:     b_k10_HSBC_s90 + b_k10_VOD_s90 + b_k10_BP_s90
                                 + b_k10_GSK_s90 + b_k10_AZN_s90
                                 + b_k10_RIO_s90 + b_k10_BG_s90
                                 + b_k10_TSCO_s90 + b_k10_AAL_s90
                                 + b_k10_PRU_s90 - s_k10_HSBC_s90
                                 - s_k10_VOD_s90 - s_k10_BP_s90 - s_k10_GSK_s90
                                 - s_k10_AZN_s90 - s_k10_RIO_s90 - s_k10_BG_s90
                                 - s_k10_TSCO_s90 - s_k10_AAL_s90
                                 - s_k10_PRU_s90  = 0
 k10_s91_balance_constraint:     b_k10_HSBC_s91 + b_k10_VOD_s91 + b_k10_BP_s91
                                 + b_k10_GSK_s91 + b_k10_AZN_s91
                                 + b_k10_RIO_s91 + b_k10_BG_s91
                                 + b_k10_TSCO_s91 + b_k10_AAL_s91
                                 + b_k10_PRU_s91 - s_k10_HSBC_s91
                                 - s_k10_VOD_s91 - s_k10_BP_s91 - s_k10_GSK_s91
                                 - s_k10_AZN_s91 - s_k10_RIO_s91 - s_k10_BG_s91
                                 - s_k10_TSCO_s91 - s_k10_AAL_s91
                                 - s_k10_PRU_s91  = 0
 k10_s92_balance_constraint:     b_k10_HSBC_s92 + b_k10_VOD_s92 + b_k10_BP_s92
                                 + b_k10_GSK_s92 + b_k10_AZN_s92
                                 + b_k10_RIO_s92 + b_k10_BG_s92
                                 + b_k10_TSCO_s92 + b_k10_AAL_s92
                                 + b_k10_PRU_s92 - s_k10_HSBC_s92
                                 - s_k10_VOD_s92 - s_k10_BP_s92 - s_k10_GSK_s92
                                 - s_k10_AZN_s92 - s_k10_RIO_s92 - s_k10_BG_s92
                                 - s_k10_TSCO_s92 - s_k10_AAL_s92
                                 - s_k10_PRU_s92  = 0
 k10_s93_balance_constraint:     b_k10_HSBC_s93 + b_k10_VOD_s93 + b_k10_BP_s93
                                 + b_k10_GSK_s93 + b_k10_AZN_s93
                                 + b_k10_RIO_s93 + b_k10_BG_s93
                                 + b_k10_TSCO_s93 + b_k10_AAL_s93
                                 + b_k10_PRU_s93 - s_k10_HSBC_s93
                                 - s_k10_VOD_s93 - s_k10_BP_s93 - s_k10_GSK_s93
                                 - s_k10_AZN_s93 - s_k10_RIO_s93 - s_k10_BG_s93
                                 - s_k10_TSCO_s93 - s_k10_AAL_s93
                                 - s_k10_PRU_s93  = 0
 k10_s94_balance_constraint:     b_k10_HSBC_s94 + b_k10_VOD_s94 + b_k10_BP_s94
                                 + b_k10_GSK_s94 + b_k10_AZN_s94
                                 + b_k10_RIO_s94 + b_k10_BG_s94
                                 + b_k10_TSCO_s94 + b_k10_AAL_s94
                                 + b_k10_PRU_s94 - s_k10_HSBC_s94
                                 - s_k10_VOD_s94 - s_k10_BP_s94 - s_k10_GSK_s94
                                 - s_k10_AZN_s94 - s_k10_RIO_s94 - s_k10_BG_s94
                                 - s_k10_TSCO_s94 - s_k10_AAL_s94
                                 - s_k10_PRU_s94  = 0
 k10_s95_balance_constraint:     b_k10_HSBC_s95 + b_k10_VOD_s95 + b_k10_BP_s95
                                 + b_k10_GSK_s95 + b_k10_AZN_s95
                                 + b_k10_RIO_s95 + b_k10_BG_s95
                                 + b_k10_TSCO_s95 + b_k10_AAL_s95
                                 + b_k10_PRU_s95 - s_k10_HSBC_s95
                                 - s_k10_VOD_s95 - s_k10_BP_s95 - s_k10_GSK_s95
                                 - s_k10_AZN_s95 - s_k10_RIO_s95 - s_k10_BG_s95
                                 - s_k10_TSCO_s95 - s_k10_AAL_s95
                                 - s_k10_PRU_s95  = 0
 k10_s96_balance_constraint:     b_k10_HSBC_s96 + b_k10_VOD_s96 + b_k10_BP_s96
                                 + b_k10_GSK_s96 + b_k10_AZN_s96
                                 + b_k10_RIO_s96 + b_k10_BG_s96
                                 + b_k10_TSCO_s96 + b_k10_AAL_s96
                                 + b_k10_PRU_s96 - s_k10_HSBC_s96
                                 - s_k10_VOD_s96 - s_k10_BP_s96 - s_k10_GSK_s96
                                 - s_k10_AZN_s96 - s_k10_RIO_s96 - s_k10_BG_s96
                                 - s_k10_TSCO_s96 - s_k10_AAL_s96
                                 - s_k10_PRU_s96  = 0
 k10_s97_balance_constraint:     b_k10_HSBC_s97 + b_k10_VOD_s97 + b_k10_BP_s97
                                 + b_k10_GSK_s97 + b_k10_AZN_s97
                                 + b_k10_RIO_s97 + b_k10_BG_s97
                                 + b_k10_TSCO_s97 + b_k10_AAL_s97
                                 + b_k10_PRU_s97 - s_k10_HSBC_s97
                                 - s_k10_VOD_s97 - s_k10_BP_s97 - s_k10_GSK_s97
                                 - s_k10_AZN_s97 - s_k10_RIO_s97 - s_k10_BG_s97
                                 - s_k10_TSCO_s97 - s_k10_AAL_s97
                                 - s_k10_PRU_s97  = 0
 k10_s98_balance_constraint:     b_k10_HSBC_s98 + b_k10_VOD_s98 + b_k10_BP_s98
                                 + b_k10_GSK_s98 + b_k10_AZN_s98
                                 + b_k10_RIO_s98 + b_k10_BG_s98
                                 + b_k10_TSCO_s98 + b_k10_AAL_s98
                                 + b_k10_PRU_s98 - s_k10_HSBC_s98
                                 - s_k10_VOD_s98 - s_k10_BP_s98 - s_k10_GSK_s98
                                 - s_k10_AZN_s98 - s_k10_RIO_s98 - s_k10_BG_s98
                                 - s_k10_TSCO_s98 - s_k10_AAL_s98
                                 - s_k10_PRU_s98  = 0
 k10_s99_balance_constraint:     b_k10_HSBC_s99 + b_k10_VOD_s99 + b_k10_BP_s99
                                 + b_k10_GSK_s99 + b_k10_AZN_s99
                                 + b_k10_RIO_s99 + b_k10_BG_s99
                                 + b_k10_TSCO_s99 + b_k10_AAL_s99
                                 + b_k10_PRU_s99 - s_k10_HSBC_s99
                                 - s_k10_VOD_s99 - s_k10_BP_s99 - s_k10_GSK_s99
                                 - s_k10_AZN_s99 - s_k10_RIO_s99 - s_k10_BG_s99
                                 - s_k10_TSCO_s99 - s_k10_AAL_s99
                                 - s_k10_PRU_s99  = 0
 k10_wcvar_constraint:           wcvar - var_10 - 1.69250098602601 z_k10_s000
                                 - 1.81500410897398 z_k10_s001
                                 - 0.459999851765163 z_k10_s010
                                 - 0.652500182734836 z_k10_s011
                                 - 0.187500178750031 z_k10_s020
                                 - 0.907500076749968 z_k10_s021
                                 - 0.595000263833362 z_k10_s030
                                 - 0.529999998666637 z_k10_s031
                                 - 0.615000483946194 z_k10_s040
                                 - 0.175000150553805 z_k10_s041
                                 - 0.862500351250034 z_k10_s050
                                 - 0.487499963749965 z_k10_s051
                                 - 0.37750195668015 z_k10_s060
                                 - 1.22000449331985 z_k10_s061
                                 - 0.179999777875962 z_k10_s070
                                 - 0.657500642624037 z_k10_s071
                                 - 0.477499620937575 z_k10_s080
                                 - 0.262500101562425 z_k10_s081
                                 - 0.952500822250139 z_k10_s090
                                 - 0.39749949274986 z_k10_s091
                                 - 0.62250071681853 z_k10_s100
                                 - 0.56250167568147 z_k10_s101
                                 - 0.4650005596306 z_k10_s110
                                 - 0.282499833619399 z_k10_s111
                                 - 1.0324975229352 z_k10_s120
                                 - 0.4999981770648 z_k10_s121
                                 - 0.220000012210071 z_k10_s130
                                 - 0.374999858539929 z_k10_s131
                                 - 1.43000409350731 z_k10_s140
                                 - 0.372500111492685 z_k10_s141
                                 - 0.242500300932456 z_k10_s150
                                 - 0.487500123317544 z_k10_s151
                                 - 0.29250006136362 z_k10_s160
                                 - 0.20250011863638 z_k10_s161
                                 - 0.250000292982285 z_k10_s170
                                 - 0.682499112017714 z_k10_s171
                                 - 0.56249959336641 z_k10_s180
                                 - 0.19500012163359 z_k10_s181
                                 - 0.389999528005421 z_k10_s190
                                 - 0.0800000357445787 z_k10_s191
                                 - 1.24250354516117 z_k10_s200
                                 - 0.640000572338827 z_k10_s201
                                 - 0.56249912582174 z_k10_s210
                                 - 0.750000146928259 z_k10_s211
                                 - 0.560000519549104 z_k10_s220
                                 - 0.459999698200895 z_k10_s221
                                 - 2.1800039770144 z_k10_s230
                                 - 0.997501622985599 z_k10_s231
                                 - 0.424999651010203 z_k10_s240
                                 - 0.434999723739796 z_k10_s241
                                 - 0.64000026856941 z_k10_s250
                                 - 0.972501321430589 z_k10_s251
                                 - 0.604999488092103 z_k10_s260
                                 - 0.332500202907897 z_k10_s261
                                 - 1.01750075726786 z_k10_s270
                                 - 0.732499255232137 z_k10_s271
                                 - 0.484999507304413 z_k10_s280
                                 - 0.447499973195586 z_k10_s281
                                 - 0.280000011403755 z_k10_s290
                                 - 0.967499471346244 z_k10_s291
                                 - 0.505000110633988 z_k10_s300
                                 - 0.324999709616011 z_k10_s301
                                 - 0.274999864866621 z_k10_s310
                                 - 0.287499730883378 z_k10_s311
                                 - 0.589999636118787 z_k10_s320
                                 - 0.459999271381212 z_k10_s321
                                 - 0.504998772257705 z_k10_s330
                                 - 1.73249900024229 z_k10_s331
                                 - 0.130000207266038 z_k10_s340
                                 - 0.340000108233962 z_k10_s341
                                 - 0.410000045499896 z_k10_s350
                                 - 0.134999709250103 z_k10_s351
                                 - 0.207500254631124 z_k10_s360
                                 - 0.307499893368876 z_k10_s361
                                 - 0.397499759909331 z_k10_s370
                                 - 0.422499909840669 z_k10_s371
                                 - 0.165000101655617 z_k10_s380
                                 - 0.437500096094382 z_k10_s381
                                 - 0.267499981906384 z_k10_s390
                                 - 0.547499612593615 z_k10_s391
                                 - 0.4249994196228 z_k10_s400
                                 - 0.635000180377199 z_k10_s401
                                 - 0.382499669070384 z_k10_s410
                                 - 0.257500044929616 z_k10_s411
                                 - 0.18250004656626 z_k10_s420
                                 - 0.23250009343374 z_k10_s421
                                 - 0.375001200504 z_k10_s430
                                 - 0.517501799496 z_k10_s431
                                 - 0.314999879878272 z_k10_s440
                                 - 0.259999992121728 z_k10_s441
                                 - 0.66250035235896 z_k10_s450
                                 - 0.31500068764104 z_k10_s451
                                 - 0.842498924473799 z_k10_s460
                                 - 0.2974997555262 z_k10_s461
                                 - 0.337500407426952 z_k10_s470
                                 - 0.174999888573048 z_k10_s471
                                 - 0.915000797338379 z_k10_s480
                                 - 0.32500066266162 z_k10_s481
                                 - 0.289999911091968 z_k10_s490
                                 - 0.117500216908032 z_k10_s491
                                 - 0.4649997433095 z_k10_s500
                                 - 0.2449994066905 z_k10_s501
                                 - 0.182500205157715 z_k10_s510
                                 - 0.0949999623422849 z_k10_s511
                                 - 0.129999903494829 z_k10_s520
                                 - 0.112500131005171 z_k10_s521
                                 - 0.290000126618433 z_k10_s530
                                 - 0.0899999868815669 z_k10_s531
                                 - 0.0999999139830614 z_k10_s540
                                 - 0.194999876516938 z_k10_s541
                                 - 0.62000200712516 z_k10_s550
                                 - 0.14500001287484 z_k10_s551
                                 - 0.227499833583363 z_k10_s560
                                 - 0.0875000329166369 z_k10_s561
                                 - 0.167499698164875 z_k10_s570
                                 - 0.302499426835125 z_k10_s571
                                 - 0.42250165550019 z_k10_s580
                                 - 0.10250033949981 z_k10_s581
                                 - 0.30749988958792 z_k10_s590
                                 - 0.14750006541208 z_k10_s591
                                 - 0.2925003177602 z_k10_s600
                                 - 0.1399998822398 z_k10_s601
                                 - 0.412500396596475 z_k10_s610
                                 - 0.209999828403525 z_k10_s611
                                 - 0.0849998150522499 z_k10_s620
                                 - 0.39250040994775 z_k10_s621
                                 - 0.365000250312 z_k10_s630
                                 - 0.915001749687999 z_k10_s631
                                 - 0.167499928138275 z_k10_s640
                                 - 0.185000146861725 z_k10_s641
                                 - 0.667501184582999 z_k10_s650
                                 - 0.775002315416999 z_k10_s651
                                 - 0.44499978 z_k10_s660
                                 - 0.20750022 z_k10_s661
                                 - 0.157499919918425 z_k10_s670
                                 - 0.302500255081575 z_k10_s671
                                 - 0.627498711399999 z_k10_s680
                                 - 0.4249982886 z_k10_s681
                                 - 0.21250001513085 z_k10_s690
                                 - 0.26500020986915 z_k10_s691
                                 - 0.579999966319125 z_k10_s700
                                 - 0.187500319930875 z_k10_s701
                                 - 0.13999983000003 z_k10_s710
                                 - 0.595000012499969 z_k10_s711
                                 - 0.195000126818612 z_k10_s720
                                 - 0.815002135681387 z_k10_s721
                                 - 0.247500161396565 z_k10_s730
                                 - 0.114999873603435 z_k10_s731
                                 - 0.46749989250904 z_k10_s740
                                 - 0.22250018499096 z_k10_s741
                                 - 0.3625002646209 z_k10_s750
                                 - 0.1649999603791 z_k10_s751
                                 - 0.687501322926599 z_k10_s760
                                 - 0.4400010270734 z_k10_s761
                                 - 0.159999880833292 z_k10_s770
                                 - 0.320000241666707 z_k10_s771
                                 - 0.684999620904874 z_k10_s780
                                 - 0.764998941595124 z_k10_s781
                                 - 0.46749988910164 z_k10_s790
                                 - 0.27000029089836 z_k10_s791
                                 - 1.34499863511912 z_k10_s800
                                 - 0.865000157380877 z_k10_s801
                                 - 2.02000189702082 z_k10_s810
                                 - 2.26250302047918 z_k10_s811
                                 - 0.619999465041134 z_k10_s820
                                 - 0.474999678458865 z_k10_s821
                                 - 0.430000300818813 z_k10_s830
                                 - 0.942500236431186 z_k10_s831
                                 - 0.909999436054222 z_k10_s840
                                 - 0.197500455695777 z_k10_s841
                                 - 0.752502713682134 z_k10_s850
                                 - 1.11000329131786 z_k10_s851
                                 - 0.802499545865867 z_k10_s860
                                 - 0.357500375884132 z_k10_s861
                                 - 0.744999593545825 z_k10_s870
                                 - 0.292500258204174 z_k10_s871
                                 - 1.44000493629082 z_k10_s880
                                 - 0.645003666209182 z_k10_s881
                                 - 0.89249981340968 z_k10_s890
                                 - 0.187499815590319 z_k10_s891
                                 - 0.28250061437186 z_k10_s900
                                 - 0.822502258128139 z_k10_s901
                                 - 0.222498787773032 z_k10_s910
                                 - 0.767497909726967 z_k10_s911
                                 - 0.36250023435625 z_k10_s920
                                 - 0.13750008889375 z_k10_s921
                                 - 0.152500235272959 z_k10_s930
                                 - 0.175000167977041 z_k10_s931
                                 - 0.462499747746432 z_k10_s940
                                 - 0.247500356253568 z_k10_s941
                                 - 1.03500291427973 z_k10_s950
                                 - 0.39750129572027 z_k10_s951
                                 - 0.772501872731379 z_k10_s960
                                 - 0.20250011226862 z_k10_s961
                                 - 0.950001527124099 z_k10_s970
                                 - 0.580001012875899 z_k10_s971
                                 - 0.367500632001072 z_k10_s980
                                 - 0.872501365498927 z_k10_s981
                                 - 0.342499790168976 z_k10_s990
                                 - 0.190000287831024 z_k10_s991 >= 0
 z_k10_s000_constraint:          var_10 + z_k10_s000 + 0.943611 w_k10_HSBC_s00
                                 + 0.980878 w_k10_VOD_s00
                                 + 0.876518 w_k10_BP_s00
                                 + 0.909853 w_k10_GSK_s00
                                 + 0.936299 w_k10_AZN_s00
                                 + 0.825285 w_k10_RIO_s00
                                 + 0.952209 w_k10_BG_s00
                                 + 0.835925 w_k10_TSCO_s00
                                 + 1.06229 w_k10_AAL_s00
                                 + 0.880491 w_k10_PRU_s00 >= 1
 z_k10_s001_constraint:          var_10 + z_k10_s001 + 1.02954 w_k10_HSBC_s00
                                 + 1.04973 w_k10_VOD_s00 + 1.12539 w_k10_BP_s00
                                 + 1.02084 w_k10_GSK_s00
                                 + 0.908133 w_k10_AZN_s00
                                 + 1.14573 w_k10_RIO_s00 + 1.01112 w_k10_BG_s00
                                 + 1.15286 w_k10_TSCO_s00
                                 + 1.01266 w_k10_AAL_s00
                                 + 1.02624 w_k10_PRU_s00 >= 1
 z_k10_s010_constraint:          var_10 + z_k10_s010 + 0.918278 w_k10_HSBC_s01
                                 + 1.06372 w_k10_VOD_s01
                                 + 0.914306 w_k10_BP_s01
                                 + 0.956805 w_k10_GSK_s01
                                 + 0.988257 w_k10_AZN_s01
                                 + 0.972788 w_k10_RIO_s01
                                 + 0.929386 w_k10_BG_s01
                                 + 0.828606 w_k10_TSCO_s01
                                 + 1.10179 w_k10_AAL_s01 + 1.0786 w_k10_PRU_s01
                                 >= 1
 z_k10_s011_constraint:          var_10 + z_k10_s011 + 1.02665 w_k10_HSBC_s01
                                 + 0.966475 w_k10_VOD_s01
                                 + 0.985914 w_k10_BP_s01
                                 + 0.945743 w_k10_GSK_s01
                                 + 0.997097 w_k10_AZN_s01
                                 + 1.01757 w_k10_RIO_s01
                                 + 0.995722 w_k10_BG_s01
                                 + 1.02296 w_k10_TSCO_s01
                                 + 0.916212 w_k10_AAL_s01
                                 + 0.896369 w_k10_PRU_s01 >= 1
 z_k10_s020_constraint:          var_10 + z_k10_s020 + 0.844393 w_k10_HSBC_s02
                                 + 0.962354 w_k10_VOD_s02
                                 + 0.783912 w_k10_BP_s02
                                 + 0.855395 w_k10_GSK_s02
                                 + 1.05697 w_k10_AZN_s02
                                 + 0.787863 w_k10_RIO_s02
                                 + 0.902923 w_k10_BG_s02
                                 + 0.867163 w_k10_TSCO_s02
                                 + 1.04079 w_k10_AAL_s02
                                 + 1.00909 w_k10_PRU_s02 >= 1
 z_k10_s021_constraint:          var_10 + z_k10_s021 + 0.982235 w_k10_HSBC_s02
                                 + 1.02657 w_k10_VOD_s02 + 1.00577 w_k10_BP_s02
                                 + 1.04699 w_k10_GSK_s02
                                 + 1.12304 w_k10_AZN_s02
                                 + 0.997451 w_k10_RIO_s02
                                 + 1.02545 w_k10_BG_s02
                                 + 0.967256 w_k10_TSCO_s02
                                 + 1.01735 w_k10_AAL_s02
                                 + 1.05603 w_k10_PRU_s02 >= 1
 z_k10_s030_constraint:          var_10 + z_k10_s030 + 0.824852 w_k10_HSBC_s03
                                 + 0.870772 w_k10_VOD_s03
                                 + 0.852664 w_k10_BP_s03
                                 + 0.860508 w_k10_GSK_s03
                                 + 0.921846 w_k10_AZN_s03
                                 + 0.869909 w_k10_RIO_s03
                                 + 0.999375 w_k10_BG_s03
                                 + 0.921451 w_k10_TSCO_s03
                                 + 0.972073 w_k10_AAL_s03
                                 + 0.992917 w_k10_PRU_s03 >= 1
 z_k10_s031_constraint:          var_10 + z_k10_s031 + 0.8971 w_k10_HSBC_s03
                                 + 1.104 w_k10_VOD_s03 + 1.16739 w_k10_BP_s03
                                 + 1.08369 w_k10_GSK_s03
                                 + 1.15093 w_k10_AZN_s03
                                 + 1.03766 w_k10_RIO_s03 + 1.0357 w_k10_BG_s03
                                 + 1.14592 w_k10_TSCO_s03
                                 + 1.02598 w_k10_AAL_s03
                                 + 1.10486 w_k10_PRU_s03 >= 1
 z_k10_s040_constraint:          var_10 + z_k10_s040 + 0.891078 w_k10_HSBC_s04
                                 + 0.945662 w_k10_VOD_s04
                                 + 1.04895 w_k10_BP_s04
                                 + 0.991678 w_k10_GSK_s04
                                 + 0.979045 w_k10_AZN_s04
                                 + 0.975785 w_k10_RIO_s04
                                 + 0.944765 w_k10_BG_s04
                                 + 0.961618 w_k10_TSCO_s04
                                 + 1.02775 w_k10_AAL_s04
                                 + 1.09503 w_k10_PRU_s04 >= 1
 z_k10_s041_constraint:          var_10 + z_k10_s041 + 1.07383 w_k10_HSBC_s04
                                 + 0.947789 w_k10_VOD_s04
                                 + 1.15347 w_k10_BP_s04 + 1.12331 w_k10_GSK_s04
                                 + 1.12251 w_k10_AZN_s04
                                 + 1.11887 w_k10_RIO_s04 + 1.03965 w_k10_BG_s04
                                 + 1.31591 w_k10_TSCO_s04
                                 + 0.92578 w_k10_AAL_s04
                                 + 1.06075 w_k10_PRU_s04 >= 1
 z_k10_s050_constraint:          var_10 + z_k10_s050 + 0.880961 w_k10_HSBC_s05
                                 + 0.973499 w_k10_VOD_s05
                                 + 0.982902 w_k10_BP_s05
                                 + 1.07858 w_k10_GSK_s05
                                 + 1.00124 w_k10_AZN_s05
                                 + 0.974953 w_k10_RIO_s05
                                 + 1.02842 w_k10_BG_s05
                                 + 0.891231 w_k10_TSCO_s05
                                 + 0.978465 w_k10_AAL_s05
                                 + 0.901669 w_k10_PRU_s05 >= 1
 z_k10_s051_constraint:          var_10 + z_k10_s051 + 1.11936 w_k10_HSBC_s05
                                 + 1.01506 w_k10_VOD_s05 + 1.13471 w_k10_BP_s05
                                 + 1.09554 w_k10_GSK_s05
                                 + 1.06549 w_k10_AZN_s05
                                 + 1.08014 w_k10_RIO_s05 + 1.01699 w_k10_BG_s05
                                 + 1.15981 w_k10_TSCO_s05
                                 + 1.01458 w_k10_AAL_s05
                                 + 1.08301 w_k10_PRU_s05 >= 1
 z_k10_s060_constraint:          var_10 + z_k10_s060 + 0.942437 w_k10_HSBC_s06
                                 + 0.949364 w_k10_VOD_s06
                                 + 1.18815 w_k10_BP_s06 + 1.21687 w_k10_GSK_s06
                                 + 1.0556 w_k10_AZN_s06 + 1.10851 w_k10_RIO_s06
                                 + 0.942594 w_k10_BG_s06
                                 + 1.20345 w_k10_TSCO_s06
                                 + 1.06664 w_k10_AAL_s06
                                 + 1.16742 w_k10_PRU_s06 >= 1
 z_k10_s061_constraint:          var_10 + z_k10_s061 + 0.941121 w_k10_HSBC_s06
                                 + 0.936526 w_k10_VOD_s06
                                 + 0.922898 w_k10_BP_s06
                                 + 0.924493 w_k10_GSK_s06
                                 + 0.954984 w_k10_AZN_s06
                                 + 0.904544 w_k10_RIO_s06
                                 + 0.970539 w_k10_BG_s06
                                 + 0.889127 w_k10_TSCO_s06
                                 + 0.97664 w_k10_AAL_s06
                                 + 0.979897 w_k10_PRU_s06 >= 1
 z_k10_s070_constraint:          var_10 + z_k10_s070 + 1.23757 w_k10_HSBC_s07
                                 + 1.15629 w_k10_VOD_s07 + 1.39499 w_k10_BP_s07
                                 + 1.13802 w_k10_GSK_s07
                                 + 1.11166 w_k10_AZN_s07
                                 + 1.22172 w_k10_RIO_s07 + 1.02995 w_k10_BG_s07
                                 + 1.31343 w_k10_TSCO_s07
                                 + 0.933625 w_k10_AAL_s07
                                 + 1.0608 w_k10_PRU_s07 >= 1
 z_k10_s071_constraint:          var_10 + z_k10_s071 + 0.926409 w_k10_HSBC_s07
                                 + 1.05902 w_k10_VOD_s07
                                 + 0.937853 w_k10_BP_s07
                                 + 1.01883 w_k10_GSK_s07
                                 + 1.15648 w_k10_AZN_s07
                                 + 0.986997 w_k10_RIO_s07
                                 + 0.998149 w_k10_BG_s07
                                 + 0.83664 w_k10_TSCO_s07
                                 + 1.01229 w_k10_AAL_s07
                                 + 1.00674 w_k10_PRU_s07 >= 1
 z_k10_s080_constraint:          var_10 + z_k10_s080 + 0.870654 w_k10_HSBC_s08
                                 + 0.979146 w_k10_VOD_s08
                                 + 0.915587 w_k10_BP_s08
                                 + 1.08193 w_k10_GSK_s08
                                 + 1.05183 w_k10_AZN_s08
                                 + 0.920162 w_k10_RIO_s08
                                 + 0.965047 w_k10_BG_s08
                                 + 0.988185 w_k10_TSCO_s08
                                 + 0.988238 w_k10_AAL_s08
                                 + 0.977264 w_k10_PRU_s08 >= 1
 z_k10_s081_constraint:          var_10 + z_k10_s081 + 1.04172 w_k10_HSBC_s08
                                 + 1.02231 w_k10_VOD_s08 + 1.19407 w_k10_BP_s08
                                 + 1.23228 w_k10_GSK_s08
                                 + 1.07179 w_k10_AZN_s08
                                 + 1.15305 w_k10_RIO_s08 + 1.04177 w_k10_BG_s08
                                 + 1.31999 w_k10_TSCO_s08
                                 + 0.972116 w_k10_AAL_s08
                                 + 0.865735 w_k10_PRU_s08 >= 1
 z_k10_s090_constraint:          var_10 + z_k10_s090 + 0.844823 w_k10_HSBC_s09
                                 + 0.985269 w_k10_VOD_s09
                                 + 0.96804 w_k10_BP_s09 + 1.00888 w_k10_GSK_s09
                                 + 0.966727 w_k10_AZN_s09
                                 + 0.96148 w_k10_RIO_s09 + 1.00895 w_k10_BG_s09
                                 + 0.916101 w_k10_TSCO_s09
                                 + 1.04541 w_k10_AAL_s09
                                 + 1.00028 w_k10_PRU_s09 >= 1
 z_k10_s091_constraint:          var_10 + z_k10_s091 + 1.16831 w_k10_HSBC_s09
                                 + 1.10763 w_k10_VOD_s09 + 1.24627 w_k10_BP_s09
                                 + 1.06299 w_k10_GSK_s09
                                 + 1.08042 w_k10_AZN_s09
                                 + 1.14576 w_k10_RIO_s09 + 1.01574 w_k10_BG_s09
                                 + 1.27781 w_k10_TSCO_s09
                                 + 1.00359 w_k10_AAL_s09 + 1.1762 w_k10_PRU_s09
                                 >= 1
 z_k10_s100_constraint:          var_10 + z_k10_s100 + 0.831287 w_k10_HSBC_s10
                                 + 0.998592 w_k10_VOD_s10
                                 + 0.950881 w_k10_BP_s10
                                 + 0.986771 w_k10_GSK_s10
                                 + 1.01166 w_k10_AZN_s10
                                 + 0.934054 w_k10_RIO_s10
                                 + 1.02475 w_k10_BG_s10
                                 + 0.932145 w_k10_TSCO_s10
                                 + 0.968834 w_k10_AAL_s10
                                 + 0.909309 w_k10_PRU_s10 >= 1
 z_k10_s101_constraint:          var_10 + z_k10_s101 + 1.1406 w_k10_HSBC_s10
                                 + 1.11376 w_k10_VOD_s10 + 1.23709 w_k10_BP_s10
                                 + 1.04886 w_k10_GSK_s10
                                 + 0.955176 w_k10_AZN_s10
                                 + 1.15902 w_k10_RIO_s10 + 1.07125 w_k10_BG_s10
                                 + 1.15759 w_k10_TSCO_s10
                                 + 1.10669 w_k10_AAL_s10
                                 + 1.05141 w_k10_PRU_s10 >= 1
 z_k10_s110_constraint:          var_10 + z_k10_s110 + 0.991163 w_k10_HSBC_s11
                                 + 1.00026 w_k10_VOD_s11 + 1.03023 w_k10_BP_s11
                                 + 1.07872 w_k10_GSK_s11
                                 + 1.07973 w_k10_AZN_s11
                                 + 1.05445 w_k10_RIO_s11
                                 + 0.995195 w_k10_BG_s11
                                 + 1.00858 w_k10_TSCO_s11
                                 + 0.930836 w_k10_AAL_s11
                                 + 0.877313 w_k10_PRU_s11 >= 1
 z_k10_s111_constraint:          var_10 + z_k10_s111 + 0.936522 w_k10_HSBC_s11
                                 + 0.975614 w_k10_VOD_s11
                                 + 0.885901 w_k10_BP_s11
                                 + 1.14967 w_k10_GSK_s11
                                 + 1.10399 w_k10_AZN_s11
                                 + 0.925203 w_k10_RIO_s11
                                 + 0.911443 w_k10_BG_s11
                                 + 0.871919 w_k10_TSCO_s11
                                 + 1.08967 w_k10_AAL_s11
                                 + 0.883269 w_k10_PRU_s11 >= 1
 z_k10_s120_constraint:          var_10 + z_k10_s120 + 0.906397 w_k10_HSBC_s12
                                 + 1.09635 w_k10_VOD_s12
                                 + 0.891911 w_k10_BP_s12
                                 + 1.01404 w_k10_GSK_s12 + 1.045 w_k10_AZN_s12
                                 + 1.07085 w_k10_RIO_s12 + 1.00869 w_k10_BG_s12
                                 + 0.943564 w_k10_TSCO_s12
                                 + 1.03925 w_k10_AAL_s12
                                 + 0.928746 w_k10_PRU_s12 >= 1
 z_k10_s121_constraint:          var_10 + z_k10_s121 + 1.18676 w_k10_HSBC_s12
                                 + 1.12448 w_k10_VOD_s12 + 1.18713 w_k10_BP_s12
                                 + 1.10889 w_k10_GSK_s12
                                 + 1.08173 w_k10_AZN_s12
                                 + 1.21896 w_k10_RIO_s12 + 1.10001 w_k10_BG_s12
                                 + 1.16852 w_k10_TSCO_s12
                                 + 1.04202 w_k10_AAL_s12
                                 + 1.04432 w_k10_PRU_s12 >= 1
 z_k10_s130_constraint:          var_10 + z_k10_s130 + 1.09525 w_k10_HSBC_s13
                                 + 1.00229 w_k10_VOD_s13 + 1.14684 w_k10_BP_s13
                                 + 1.04293 w_k10_GSK_s13
                                 + 1.11892 w_k10_AZN_s13
                                 + 0.960762 w_k10_RIO_s13
                                 + 1.06655 w_k10_BG_s13
                                 + 1.22676 w_k10_TSCO_s13
                                 + 1.02674 w_k10_AAL_s13
                                 + 1.13737 w_k10_PRU_s13 >= 1
 z_k10_s131_constraint:          var_10 + z_k10_s131 + 1.07895 w_k10_HSBC_s13
                                 + 0.947073 w_k10_VOD_s13
                                 + 0.993824 w_k10_BP_s13
                                 + 0.943791 w_k10_GSK_s13
                                 + 1.01096 w_k10_AZN_s13
                                 + 0.927232 w_k10_RIO_s13
                                 + 1.01081 w_k10_BG_s13
                                 + 0.921715 w_k10_TSCO_s13
                                 + 0.960257 w_k10_AAL_s13
                                 + 1.01058 w_k10_PRU_s13 >= 1
 z_k10_s140_constraint:          var_10 + z_k10_s140 + 0.989705 w_k10_HSBC_s14
                                 + 1.1492 w_k10_VOD_s14 + 1.06952 w_k10_BP_s14
                                 + 1.13701 w_k10_GSK_s14 + 1.1236 w_k10_AZN_s14
                                 + 1.06804 w_k10_RIO_s14
                                 + 0.983319 w_k10_BG_s14
                                 + 1.11415 w_k10_TSCO_s14
                                 + 1.03535 w_k10_AAL_s14
                                 + 1.05642 w_k10_PRU_s14 >= 1
 z_k10_s141_constraint:          var_10 + z_k10_s141 + 0.88194 w_k10_HSBC_s14
                                 + 0.883309 w_k10_VOD_s14
                                 + 0.818357 w_k10_BP_s14
                                 + 0.917736 w_k10_GSK_s14
                                 + 0.941764 w_k10_AZN_s14
                                 + 0.836372 w_k10_RIO_s14
                                 + 0.989901 w_k10_BG_s14
                                 + 0.748662 w_k10_TSCO_s14
                                 + 0.981865 w_k10_AAL_s14
                                 + 0.770608 w_k10_PRU_s14 >= 1
 z_k10_s150_constraint:          var_10 + z_k10_s150 + 0.826621 w_k10_HSBC_s15
                                 + 1.0321 w_k10_VOD_s15 + 0.787475 w_k10_BP_s15
                                 + 0.94762 w_k10_GSK_s15
                                 + 1.07665 w_k10_AZN_s15
                                 + 0.866221 w_k10_RIO_s15
                                 + 0.908356 w_k10_BG_s15
                                 + 0.928556 w_k10_TSCO_s15
                                 + 0.971126 w_k10_AAL_s15
                                 + 0.988194 w_k10_PRU_s15 >= 1
 z_k10_s151_constraint:          var_10 + z_k10_s151 + 0.964566 w_k10_HSBC_s15
                                 + 0.993922 w_k10_VOD_s15
                                 + 1.06954 w_k10_BP_s15 + 1.10956 w_k10_GSK_s15
                                 + 1.11011 w_k10_AZN_s15
                                 + 1.06302 w_k10_RIO_s15 + 1.00226 w_k10_BG_s15
                                 + 1.16991 w_k10_TSCO_s15
                                 + 1.05681 w_k10_AAL_s15
                                 + 1.23471 w_k10_PRU_s15 >= 1
 z_k10_s160_constraint:          var_10 + z_k10_s160 + 0.989593 w_k10_HSBC_s16
                                 + 1.0092 w_k10_VOD_s16 + 0.998577 w_k10_BP_s16
                                 + 0.970199 w_k10_GSK_s16
                                 + 0.970923 w_k10_AZN_s16
                                 + 1.00446 w_k10_RIO_s16 + 1.04297 w_k10_BG_s16
                                 + 0.961139 w_k10_TSCO_s16
                                 + 1.03054 w_k10_AAL_s16
                                 + 1.06761 w_k10_PRU_s16 >= 1
 z_k10_s161_constraint:          var_10 + z_k10_s161 + 0.998081 w_k10_HSBC_s16
                                 + 1.03862 w_k10_VOD_s16 + 1.16485 w_k10_BP_s16
                                 + 1.07385 w_k10_GSK_s16
                                 + 1.00384 w_k10_AZN_s16
                                 + 1.10443 w_k10_RIO_s16
                                 + 0.964053 w_k10_BG_s16
                                 + 1.22458 w_k10_TSCO_s16
                                 + 1.08961 w_k10_AAL_s16
                                 + 1.23821 w_k10_PRU_s16 >= 1
 z_k10_s170_constraint:          var_10 + z_k10_s170 + 1.00598 w_k10_HSBC_s17
                                 + 0.887399 w_k10_VOD_s17
                                 + 0.840885 w_k10_BP_s17
                                 + 0.991131 w_k10_GSK_s17
                                 + 0.947924 w_k10_AZN_s17
                                 + 0.957568 w_k10_RIO_s17
                                 + 0.945527 w_k10_BG_s17
                                 + 0.853847 w_k10_TSCO_s17
                                 + 0.916831 w_k10_AAL_s17
                                 + 0.963807 w_k10_PRU_s17 >= 1
 z_k10_s171_constraint:          var_10 + z_k10_s171 + 1.06349 w_k10_HSBC_s17
                                 + 0.901028 w_k10_VOD_s17
                                 + 1.10711 w_k10_BP_s17 + 1.16859 w_k10_GSK_s17
                                 + 0.918453 w_k10_AZN_s17
                                 + 1.12309 w_k10_RIO_s17 + 1.04375 w_k10_BG_s17
                                 + 1.06079 w_k10_TSCO_s17
                                 + 1.04032 w_k10_AAL_s17
                                 + 1.17274 w_k10_PRU_s17 >= 1
 z_k10_s180_constraint:          var_10 + z_k10_s180 + 0.973442 w_k10_HSBC_s18
                                 + 1.11615 w_k10_VOD_s18
                                 + 0.918856 w_k10_BP_s18
                                 + 1.00107 w_k10_GSK_s18
                                 + 1.05006 w_k10_AZN_s18
                                 + 1.10209 w_k10_RIO_s18
                                 + 0.984979 w_k10_BG_s18
                                 + 0.909164 w_k10_TSCO_s18
                                 + 1.00572 w_k10_AAL_s18
                                 + 1.16897 w_k10_PRU_s18 >= 1
 z_k10_s181_constraint:          var_10 + z_k10_s181 + 0.879187 w_k10_HSBC_s18
                                 + 0.854138 w_k10_VOD_s18 + 1.0239 w_k10_BP_s18
                                 + 0.989335 w_k10_GSK_s18
                                 + 1.02242 w_k10_AZN_s18
                                 + 0.95131 w_k10_RIO_s18 + 1.10846 w_k10_BG_s18
                                 + 0.933173 w_k10_TSCO_s18
                                 + 1.11414 w_k10_AAL_s18
                                 + 0.968803 w_k10_PRU_s18 >= 1
 z_k10_s190_constraint:          var_10 + z_k10_s190 + 1.09798 w_k10_HSBC_s19
                                 + 0.945182 w_k10_VOD_s19
                                 + 0.968863 w_k10_BP_s19
                                 + 0.913902 w_k10_GSK_s19
                                 + 0.945588 w_k10_AZN_s19
                                 + 1.01064 w_k10_RIO_s19 + 1.02089 w_k10_BG_s19
                                 + 0.930057 w_k10_TSCO_s19
                                 + 0.951129 w_k10_AAL_s19
                                 + 0.943994 w_k10_PRU_s19 >= 1
 z_k10_s191_constraint:          var_10 + z_k10_s191 + 1.12491 w_k10_HSBC_s19
                                 + 1.05182 w_k10_VOD_s19 + 1.30436 w_k10_BP_s19
                                 + 1.15947 w_k10_GSK_s19
                                 + 0.986828 w_k10_AZN_s19
                                 + 1.29213 w_k10_RIO_s19 + 1.04586 w_k10_BG_s19
                                 + 1.15232 w_k10_TSCO_s19
                                 + 0.97068 w_k10_AAL_s19
                                 + 0.977609 w_k10_PRU_s19 >= 1
 z_k10_s200_constraint:          var_10 + z_k10_s200 + 0.943021 w_k10_HSBC_s20
                                 + 0.925487 w_k10_VOD_s20
                                 + 0.912991 w_k10_BP_s20
                                 + 0.955461 w_k10_GSK_s20
                                 + 0.86338 w_k10_AZN_s20
                                 + 0.969541 w_k10_RIO_s20
                                 + 0.983449 w_k10_BG_s20
                                 + 0.826203 w_k10_TSCO_s20
                                 + 1.03311 w_k10_AAL_s20
                                 + 0.980296 w_k10_PRU_s20 >= 1
 z_k10_s201_constraint:          var_10 + z_k10_s201 + 1.10892 w_k10_HSBC_s20
                                 + 1.0799 w_k10_VOD_s20 + 1.21997 w_k10_BP_s20
                                 + 1.02587 w_k10_GSK_s20
                                 + 0.993753 w_k10_AZN_s20
                                 + 1.11666 w_k10_RIO_s20 + 1.06481 w_k10_BG_s20
                                 + 1.17345 w_k10_TSCO_s20
                                 + 0.964268 w_k10_AAL_s20
                                 + 0.930044 w_k10_PRU_s20 >= 1
 z_k10_s210_constraint:          var_10 + z_k10_s210 + 0.976267 w_k10_HSBC_s21
                                 + 0.955943 w_k10_VOD_s21
                                 + 0.820734 w_k10_BP_s21
                                 + 0.854677 w_k10_GSK_s21
                                 + 0.941018 w_k10_AZN_s21
                                 + 0.919603 w_k10_RIO_s21
                                 + 0.953763 w_k10_BG_s21
                                 + 0.853378 w_k10_TSCO_s21
                                 + 0.996921 w_k10_AAL_s21
                                 + 0.96498 w_k10_PRU_s21 >= 1
 z_k10_s211_constraint:          var_10 + z_k10_s211 + 1.03315 w_k10_HSBC_s21
                                 + 1.14573 w_k10_VOD_s21 + 1.09924 w_k10_BP_s21
                                 + 1.09581 w_k10_GSK_s21
                                 + 1.09229 w_k10_AZN_s21
                                 + 1.12783 w_k10_RIO_s21 + 1.03803 w_k10_BG_s21
                                 + 1.03197 w_k10_TSCO_s21
                                 + 1.03701 w_k10_AAL_s21
                                 + 1.03622 w_k10_PRU_s21 >= 1
 z_k10_s220_constraint:          var_10 + z_k10_s220 + 0.865825 w_k10_HSBC_s22
                                 + 1.04557 w_k10_VOD_s22 + 1.12142 w_k10_BP_s22
                                 + 1.07172 w_k10_GSK_s22
                                 + 0.992573 w_k10_AZN_s22
                                 + 1.10839 w_k10_RIO_s22 + 1.02748 w_k10_BG_s22
                                 + 1.07372 w_k10_TSCO_s22
                                 + 1.0943 w_k10_AAL_s22 + 1.12723 w_k10_PRU_s22
                                 >= 1
 z_k10_s221_constraint:          var_10 + z_k10_s221 + 0.934019 w_k10_HSBC_s22
                                 + 0.883037 w_k10_VOD_s22
                                 + 0.919517 w_k10_BP_s22
                                 + 0.931018 w_k10_GSK_s22
                                 + 0.88777 w_k10_AZN_s22
                                 + 0.945359 w_k10_RIO_s22
                                 + 1.04492 w_k10_BG_s22
                                 + 0.923227 w_k10_TSCO_s22
                                 + 0.915861 w_k10_AAL_s22
                                 + 0.85862 w_k10_PRU_s22 >= 1
 z_k10_s230_constraint:          var_10 + z_k10_s230 + 0.979565 w_k10_HSBC_s23
                                 + 1.08635 w_k10_VOD_s23
                                 + 0.905367 w_k10_BP_s23
                                 + 1.04893 w_k10_GSK_s23
                                 + 1.05649 w_k10_AZN_s23
                                 + 0.964712 w_k10_RIO_s23 + 0.9866 w_k10_BG_s23
                                 + 0.871887 w_k10_TSCO_s23
                                 + 1.01736 w_k10_AAL_s23
                                 + 0.982688 w_k10_PRU_s23 >= 1
 z_k10_s231_constraint:          var_10 + z_k10_s231 + 1.20156 w_k10_HSBC_s23
                                 + 0.934512 w_k10_VOD_s23
                                 + 1.12551 w_k10_BP_s23 + 1.08025 w_k10_GSK_s23
                                 + 0.928259 w_k10_AZN_s23
                                 + 1.26516 w_k10_RIO_s23 + 1.10648 w_k10_BG_s23
                                 + 1.1242 w_k10_TSCO_s23
                                 + 0.991094 w_k10_AAL_s23
                                 + 1.15172 w_k10_PRU_s23 >= 1
 z_k10_s240_constraint:          var_10 + z_k10_s240 + 0.910133 w_k10_HSBC_s24
                                 + 0.910705 w_k10_VOD_s24
                                 + 0.894426 w_k10_BP_s24
                                 + 0.86835 w_k10_GSK_s24
                                 + 0.953472 w_k10_AZN_s24
                                 + 0.869174 w_k10_RIO_s24
                                 + 0.928904 w_k10_BG_s24
                                 + 0.815261 w_k10_TSCO_s24
                                 + 1.00216 w_k10_AAL_s24
                                 + 0.901727 w_k10_PRU_s24 >= 1
 z_k10_s241_constraint:          var_10 + z_k10_s241 + 1.09052 w_k10_HSBC_s24
                                 + 1.09989 w_k10_VOD_s24 + 1.14976 w_k10_BP_s24
                                 + 1.0111 w_k10_GSK_s24 + 1.02159 w_k10_AZN_s24
                                 + 1.08629 w_k10_RIO_s24 + 1.0401 w_k10_BG_s24
                                 + 1.12204 w_k10_TSCO_s24
                                 + 0.999103 w_k10_AAL_s24
                                 + 0.883282 w_k10_PRU_s24 >= 1
 z_k10_s250_constraint:          var_10 + z_k10_s250 + 1.01604 w_k10_HSBC_s25
                                 + 0.963994 w_k10_VOD_s25
                                 + 1.08329 w_k10_BP_s25 + 1.04955 w_k10_GSK_s25
                                 + 1.03425 w_k10_AZN_s25
                                 + 1.01016 w_k10_RIO_s25 + 1.00925 w_k10_BG_s25
                                 + 1.1214 w_k10_TSCO_s25
                                 + 1.05726 w_k10_AAL_s25
                                 + 1.24405 w_k10_PRU_s25 >= 1
 z_k10_s251_constraint:          var_10 + z_k10_s251 + 1.08556 w_k10_HSBC_s25
                                 + 0.858705 w_k10_VOD_s25
                                 + 0.964214 w_k10_BP_s25
                                 + 0.971685 w_k10_GSK_s25
                                 + 0.898104 w_k10_AZN_s25
                                 + 0.967628 w_k10_RIO_s25
                                 + 1.01485 w_k10_BG_s25
                                 + 0.963127 w_k10_TSCO_s25
                                 + 0.974396 w_k10_AAL_s25
                                 + 0.964262 w_k10_PRU_s25 >= 1
 z_k10_s260_constraint:          var_10 + z_k10_s260 + 1.04111 w_k10_HSBC_s26
                                 + 0.92573 w_k10_VOD_s26 + 1.16126 w_k10_BP_s26
                                 + 1.04333 w_k10_GSK_s26
                                 + 0.99074 w_k10_AZN_s26
                                 + 1.18377 w_k10_RIO_s26 + 1.05768 w_k10_BG_s26
                                 + 1.08721 w_k10_TSCO_s26
                                 + 0.982334 w_k10_AAL_s26
                                 + 1.08424 w_k10_PRU_s26 >= 1
 z_k10_s261_constraint:          var_10 + z_k10_s261 + 0.908362 w_k10_HSBC_s26
                                 + 0.902686 w_k10_VOD_s26
                                 + 0.942663 w_k10_BP_s26
                                 + 0.867078 w_k10_GSK_s26
                                 + 0.92134 w_k10_AZN_s26
                                 + 1.00799 w_k10_RIO_s26
                                 + 0.948596 w_k10_BG_s26
                                 + 0.927965 w_k10_TSCO_s26
                                 + 0.973174 w_k10_AAL_s26
                                 + 0.958809 w_k10_PRU_s26 >= 1
 z_k10_s270_constraint:          var_10 + z_k10_s270 + 1.00676 w_k10_HSBC_s27
                                 + 0.997196 w_k10_VOD_s27 + 1.1214 w_k10_BP_s27
                                 + 1.15968 w_k10_GSK_s27
                                 + 1.01068 w_k10_AZN_s27
                                 + 1.17807 w_k10_RIO_s27 + 1.04907 w_k10_BG_s27
                                 + 1.1226 w_k10_TSCO_s27
                                 + 0.861249 w_k10_AAL_s27
                                 + 1.08352 w_k10_PRU_s27 >= 1
 z_k10_s271_constraint:          var_10 + z_k10_s271 + 0.901944 w_k10_HSBC_s27
                                 + 0.908581 w_k10_VOD_s27
                                 + 0.888462 w_k10_BP_s27
                                 + 0.966805 w_k10_GSK_s27
                                 + 1.03676 w_k10_AZN_s27
                                 + 0.922295 w_k10_RIO_s27
                                 + 0.910608 w_k10_BG_s27
                                 + 0.827978 w_k10_TSCO_s27
                                 + 0.965565 w_k10_AAL_s27
                                 + 1.09566 w_k10_PRU_s27 >= 1
 z_k10_s280_constraint:          var_10 + z_k10_s280 + 1.08158 w_k10_HSBC_s28
                                 + 0.930039 w_k10_VOD_s28
                                 + 1.12149 w_k10_BP_s28 + 1.08366 w_k10_GSK_s28
                                 + 0.946508 w_k10_AZN_s28
                                 + 1.0615 w_k10_RIO_s28 + 1.06816 w_k10_BG_s28
                                 + 1.21219 w_k10_TSCO_s28
                                 + 0.984001 w_k10_AAL_s28
                                 + 0.885852 w_k10_PRU_s28 >= 1
 z_k10_s281_constraint:          var_10 + z_k10_s281 + 0.990784 w_k10_HSBC_s28
                                 + 0.95891 w_k10_VOD_s28
                                 + 0.839229 w_k10_BP_s28
                                 + 0.851185 w_k10_GSK_s28
                                 + 0.941939 w_k10_AZN_s28
                                 + 0.815532 w_k10_RIO_s28
                                 + 0.929975 w_k10_BG_s28
                                 + 0.845891 w_k10_TSCO_s28
                                 + 1.0752 w_k10_AAL_s28
                                 + 0.939684 w_k10_PRU_s28 >= 1
 z_k10_s290_constraint:          var_10 + z_k10_s290 + 1.12303 w_k10_HSBC_s29
                                 + 1.02285 w_k10_VOD_s29 + 1.18268 w_k10_BP_s29
                                 + 1.16316 w_k10_GSK_s29
                                 + 1.01257 w_k10_AZN_s29
                                 + 1.13987 w_k10_RIO_s29
                                 + 0.968209 w_k10_BG_s29
                                 + 1.12385 w_k10_TSCO_s29
                                 + 1.06518 w_k10_AAL_s29 + 1.1651 w_k10_PRU_s29
                                 >= 1
 z_k10_s291_constraint:          var_10 + z_k10_s291 + 1.00924 w_k10_HSBC_s29
                                 + 0.86022 w_k10_VOD_s29
                                 + 0.895539 w_k10_BP_s29
                                 + 0.915643 w_k10_GSK_s29
                                 + 0.999785 w_k10_AZN_s29
                                 + 0.907903 w_k10_RIO_s29
                                 + 0.985306 w_k10_BG_s29
                                 + 0.871418 w_k10_TSCO_s29
                                 + 0.969163 w_k10_AAL_s29
                                 + 0.862089 w_k10_PRU_s29 >= 1
 z_k10_s300_constraint:          var_10 + z_k10_s300 + 1.01945 w_k10_HSBC_s30
                                 + 1.00504 w_k10_VOD_s30 + 1.15736 w_k10_BP_s30
                                 + 1.21388 w_k10_GSK_s30
                                 + 1.08687 w_k10_AZN_s30
                                 + 1.02321 w_k10_RIO_s30 + 1.10945 w_k10_BG_s30
                                 + 1.2017 w_k10_TSCO_s30 + 1.0473 w_k10_AAL_s30
                                 + 0.846158 w_k10_PRU_s30 >= 1
 z_k10_s301_constraint:          var_10 + z_k10_s301 + 0.926912 w_k10_HSBC_s30
                                 + 1.02794 w_k10_VOD_s30
                                 + 0.833534 w_k10_BP_s30
                                 + 0.899051 w_k10_GSK_s30
                                 + 0.931983 w_k10_AZN_s30
                                 + 0.959683 w_k10_RIO_s30
                                 + 1.00583 w_k10_BG_s30
                                 + 0.814019 w_k10_TSCO_s30
                                 + 0.952561 w_k10_AAL_s30
                                 + 0.844716 w_k10_PRU_s30 >= 1
 z_k10_s310_constraint:          var_10 + z_k10_s310 + 0.928416 w_k10_HSBC_s31
                                 + 0.900458 w_k10_VOD_s31
                                 + 0.848515 w_k10_BP_s31
                                 + 0.931799 w_k10_GSK_s31
                                 + 0.87827 w_k10_AZN_s31
                                 + 0.969113 w_k10_RIO_s31
                                 + 0.963526 w_k10_BG_s31
                                 + 0.866713 w_k10_TSCO_s31
                                 + 0.995305 w_k10_AAL_s31
                                 + 0.981494 w_k10_PRU_s31 >= 1
 z_k10_s311_constraint:          var_10 + z_k10_s311 + 1.16553 w_k10_HSBC_s31
                                 + 1.11064 w_k10_VOD_s31 + 1.07459 w_k10_BP_s31
                                 + 1.0134 w_k10_GSK_s31 + 1.06339 w_k10_AZN_s31
                                 + 1.15656 w_k10_RIO_s31 + 1.02191 w_k10_BG_s31
                                 + 1.12404 w_k10_TSCO_s31
                                 + 1.00787 w_k10_AAL_s31
                                 + 1.10311 w_k10_PRU_s31 >= 1
 z_k10_s320_constraint:          var_10 + z_k10_s320 + 1.01419 w_k10_HSBC_s32
                                 + 0.994644 w_k10_VOD_s32
                                 + 1.10081 w_k10_BP_s32 + 1.00662 w_k10_GSK_s32
                                 + 0.959756 w_k10_AZN_s32
                                 + 1.13922 w_k10_RIO_s32
                                 + 0.954723 w_k10_BG_s32
                                 + 1.05368 w_k10_TSCO_s32
                                 + 0.906994 w_k10_AAL_s32
                                 + 1.01068 w_k10_PRU_s32 >= 1
 z_k10_s321_constraint:          var_10 + z_k10_s321 + 0.934229 w_k10_HSBC_s32
                                 + 0.851466 w_k10_VOD_s32
                                 + 1.00798 w_k10_BP_s32
                                 + 0.984247 w_k10_GSK_s32
                                 + 0.902218 w_k10_AZN_s32
                                 + 0.993663 w_k10_RIO_s32
                                 + 1.00077 w_k10_BG_s32
                                 + 0.967574 w_k10_TSCO_s32
                                 + 1.07863 w_k10_AAL_s32
                                 + 0.809839 w_k10_PRU_s32 >= 1
 z_k10_s330_constraint:          var_10 + z_k10_s330 + 1.17773 w_k10_HSBC_s33
                                 + 0.977899 w_k10_VOD_s33
                                 + 1.31596 w_k10_BP_s33 + 1.12387 w_k10_GSK_s33
                                 + 1.05124 w_k10_AZN_s33
                                 + 1.26591 w_k10_RIO_s33 + 1.05118 w_k10_BG_s33
                                 + 1.14046 w_k10_TSCO_s33
                                 + 1.01467 w_k10_AAL_s33
                                 + 1.04008 w_k10_PRU_s33 >= 1
 z_k10_s331_constraint:          var_10 + z_k10_s331 + 1.02349 w_k10_HSBC_s33
                                 + 1.04877 w_k10_VOD_s33
                                 + 0.933399 w_k10_BP_s33
                                 + 0.95142 w_k10_GSK_s33
                                 + 0.937812 w_k10_AZN_s33
                                 + 0.985404 w_k10_RIO_s33
                                 + 0.986288 w_k10_BG_s33
                                 + 1.01616 w_k10_TSCO_s33
                                 + 1.05074 w_k10_AAL_s33
                                 + 0.965213 w_k10_PRU_s33 >= 1
 z_k10_s340_constraint:          var_10 + z_k10_s340 + 0.979552 w_k10_HSBC_s34
                                 + 0.994794 w_k10_VOD_s34
                                 + 1.15796 w_k10_BP_s34 + 1.11146 w_k10_GSK_s34
                                 + 1.16131 w_k10_AZN_s34 + 1.1194 w_k10_RIO_s34
                                 + 1.09278 w_k10_BG_s34
                                 + 1.03861 w_k10_TSCO_s34
                                 + 1.08456 w_k10_AAL_s34
                                 + 1.40801 w_k10_PRU_s34 >= 1
 z_k10_s341_constraint:          var_10 + z_k10_s341 + 1.00155 w_k10_HSBC_s34
                                 + 1.06049 w_k10_VOD_s34
                                 + 0.899719 w_k10_BP_s34
                                 + 1.01265 w_k10_GSK_s34
                                 + 1.13211 w_k10_AZN_s34
                                 + 0.909267 w_k10_RIO_s34
                                 + 0.953018 w_k10_BG_s34
                                 + 0.942118 w_k10_TSCO_s34
                                 + 0.938048 w_k10_AAL_s34
                                 + 0.930836 w_k10_PRU_s34 >= 1
 z_k10_s350_constraint:          var_10 + z_k10_s350 + 1.06203 w_k10_HSBC_s35
                                 + 1.04625 w_k10_VOD_s35
                                 + 0.941889 w_k10_BP_s35
                                 + 0.799935 w_k10_GSK_s35
                                 + 0.946135 w_k10_AZN_s35
                                 + 0.971475 w_k10_RIO_s35
                                 + 0.965679 w_k10_BG_s35
                                 + 0.813142 w_k10_TSCO_s35
                                 + 0.984569 w_k10_AAL_s35
                                 + 1.05302 w_k10_PRU_s35 >= 1
 z_k10_s351_constraint:          var_10 + z_k10_s351 + 1.03165 w_k10_HSBC_s35
                                 + 1.06341 w_k10_VOD_s35 + 1.23021 w_k10_BP_s35
                                 + 1.21783 w_k10_GSK_s35
                                 + 1.05076 w_k10_AZN_s35
                                 + 1.12618 w_k10_RIO_s35 + 1.03049 w_k10_BG_s35
                                 + 1.33494 w_k10_TSCO_s35
                                 + 1.09988 w_k10_AAL_s35
                                 + 1.13815 w_k10_PRU_s35 >= 1
 z_k10_s360_constraint:          var_10 + z_k10_s360 + 1.12488 w_k10_HSBC_s36
                                 + 0.889605 w_k10_VOD_s36
                                 + 1.12982 w_k10_BP_s36
                                 + 0.926557 w_k10_GSK_s36
                                 + 1.03348 w_k10_AZN_s36
                                 + 1.10694 w_k10_RIO_s36 + 1.07638 w_k10_BG_s36
                                 + 1.16203 w_k10_TSCO_s36
                                 + 0.890302 w_k10_AAL_s36
                                 + 1.06049 w_k10_PRU_s36 >= 1
 z_k10_s361_constraint:          var_10 + z_k10_s361 + 0.877964 w_k10_HSBC_s36
                                 + 0.98308 w_k10_VOD_s36
                                 + 0.914529 w_k10_BP_s36
                                 + 1.02363 w_k10_GSK_s36
                                 + 1.09478 w_k10_AZN_s36
                                 + 0.921335 w_k10_RIO_s36
                                 + 0.990103 w_k10_BG_s36
                                 + 0.963603 w_k10_TSCO_s36
                                 + 1.05579 w_k10_AAL_s36
                                 + 0.864615 w_k10_PRU_s36 >= 1
 z_k10_s370_constraint:          var_10 + z_k10_s370 + 0.913391 w_k10_HSBC_s37
                                 + 0.841272 w_k10_VOD_s37
                                 + 0.827241 w_k10_BP_s37
                                 + 0.911765 w_k10_GSK_s37
                                 + 0.909829 w_k10_AZN_s37
                                 + 0.870121 w_k10_RIO_s37
                                 + 1.02951 w_k10_BG_s37
                                 + 0.992912 w_k10_TSCO_s37
                                 + 0.934166 w_k10_AAL_s37
                                 + 0.964885 w_k10_PRU_s37 >= 1
 z_k10_s371_constraint:          var_10 + z_k10_s371 + 1.15992 w_k10_HSBC_s37
                                 + 1.00384 w_k10_VOD_s37 + 1.10645 w_k10_BP_s37
                                 + 1.16415 w_k10_GSK_s37
                                 + 1.01682 w_k10_AZN_s37
                                 + 1.06721 w_k10_RIO_s37 + 1.02479 w_k10_BG_s37
                                 + 1.26233 w_k10_TSCO_s37
                                 + 1.01473 w_k10_AAL_s37
                                 + 1.09593 w_k10_PRU_s37 >= 1
 z_k10_s380_constraint:          var_10 + z_k10_s380 + 0.782603 w_k10_HSBC_s38
                                 + 0.934574 w_k10_VOD_s38
                                 + 0.781007 w_k10_BP_s38
                                 + 0.976412 w_k10_GSK_s38
                                 + 0.93515 w_k10_AZN_s38
                                 + 0.789425 w_k10_RIO_s38
                                 + 0.936417 w_k10_BG_s38
                                 + 0.826006 w_k10_TSCO_s38
                                 + 1.00973 w_k10_AAL_s38
                                 + 0.888853 w_k10_PRU_s38 >= 1
 z_k10_s381_constraint:          var_10 + z_k10_s381 + 1.16192 w_k10_HSBC_s38
                                 + 1.02991 w_k10_VOD_s38 + 1.01171 w_k10_BP_s38
                                 + 1.06721 w_k10_GSK_s38
                                 + 0.917947 w_k10_AZN_s38
                                 + 1.04824 w_k10_RIO_s38
                                 + 0.990912 w_k10_BG_s38
                                 + 1.00636 w_k10_TSCO_s38
                                 + 0.960915 w_k10_AAL_s38
                                 + 0.902882 w_k10_PRU_s38 >= 1
 z_k10_s390_constraint:          var_10 + z_k10_s390 + 1.03482 w_k10_HSBC_s39
                                 + 0.999392 w_k10_VOD_s39
                                 + 0.933474 w_k10_BP_s39
                                 + 0.956244 w_k10_GSK_s39
                                 + 0.927949 w_k10_AZN_s39
                                 + 1.10795 w_k10_RIO_s39 + 1.04651 w_k10_BG_s39
                                 + 0.924929 w_k10_TSCO_s39
                                 + 0.884994 w_k10_AAL_s39
                                 + 0.923302 w_k10_PRU_s39 >= 1
 z_k10_s391_constraint:          var_10 + z_k10_s391 + 1.01652 w_k10_HSBC_s39
                                 + 1.07672 w_k10_VOD_s39
                                 + 0.954779 w_k10_BP_s39
                                 + 0.93513 w_k10_GSK_s39
                                 + 0.973922 w_k10_AZN_s39
                                 + 0.977206 w_k10_RIO_s39
                                 + 1.01603 w_k10_BG_s39
                                 + 0.909988 w_k10_TSCO_s39
                                 + 1.03147 w_k10_AAL_s39
                                 + 1.05834 w_k10_PRU_s39 >= 1
 z_k10_s400_constraint:          var_10 + z_k10_s400 + 1.12921 w_k10_HSBC_s40
                                 + 1.02634 w_k10_VOD_s40 + 1.21054 w_k10_BP_s40
                                 + 1.03665 w_k10_GSK_s40
                                 + 0.992446 w_k10_AZN_s40
                                 + 1.19604 w_k10_RIO_s40 + 1.08102 w_k10_BG_s40
                                 + 1.27961 w_k10_TSCO_s40
                                 + 1.02714 w_k10_AAL_s40
                                 + 0.981703 w_k10_PRU_s40 >= 1
 z_k10_s401_constraint:          var_10 + z_k10_s401 + 0.838745 w_k10_HSBC_s40
                                 + 0.918155 w_k10_VOD_s40
                                 + 0.907641 w_k10_BP_s40
                                 + 0.980719 w_k10_GSK_s40
                                 + 0.978308 w_k10_AZN_s40
                                 + 0.983404 w_k10_RIO_s40
                                 + 0.997172 w_k10_BG_s40 + 1.001 w_k10_TSCO_s40
                                 + 0.989281 w_k10_AAL_s40
                                 + 1.11619 w_k10_PRU_s40 >= 1
 z_k10_s410_constraint:          var_10 + z_k10_s410 + 0.906884 w_k10_HSBC_s41
                                 + 1.054 w_k10_VOD_s41 + 1.09094 w_k10_BP_s41
                                 + 1.13551 w_k10_GSK_s41
                                 + 1.06434 w_k10_AZN_s41
                                 + 1.04645 w_k10_RIO_s41 + 1.06724 w_k10_BG_s41
                                 + 1.03957 w_k10_TSCO_s41
                                 + 1.0541 w_k10_AAL_s41 + 1.24501 w_k10_PRU_s41
                                 >= 1
 z_k10_s411_constraint:          var_10 + z_k10_s411 + 0.970152 w_k10_HSBC_s41
                                 + 0.939347 w_k10_VOD_s41
                                 + 0.81246 w_k10_BP_s41
                                 + 0.987662 w_k10_GSK_s41
                                 + 1.04053 w_k10_AZN_s41
                                 + 0.808753 w_k10_RIO_s41
                                 + 0.97025 w_k10_BG_s41
                                 + 0.859207 w_k10_TSCO_s41
                                 + 0.94927 w_k10_AAL_s41
                                 + 0.892606 w_k10_PRU_s41 >= 1
 z_k10_s420_constraint:          var_10 + z_k10_s420 + 1.05373 w_k10_HSBC_s42
                                 + 1.12857 w_k10_VOD_s42 + 1.18071 w_k10_BP_s42
                                 + 1.10961 w_k10_GSK_s42
                                 + 0.952674 w_k10_AZN_s42
                                 + 1.28535 w_k10_RIO_s42 + 1.05931 w_k10_BG_s42
                                 + 1.17197 w_k10_TSCO_s42
                                 + 0.969512 w_k10_AAL_s42
                                 + 1.59073 w_k10_PRU_s42 >= 1
 z_k10_s421_constraint:          var_10 + z_k10_s421 + 0.995796 w_k10_HSBC_s42
                                 + 1.10809 w_k10_VOD_s42
                                 + 0.913852 w_k10_BP_s42
                                 + 0.935322 w_k10_GSK_s42
                                 + 0.979508 w_k10_AZN_s42
                                 + 1.00572 w_k10_RIO_s42
                                 + 0.962843 w_k10_BG_s42
                                 + 0.889414 w_k10_TSCO_s42
                                 + 0.994762 w_k10_AAL_s42
                                 + 1.30481 w_k10_PRU_s42 >= 1
 z_k10_s430_constraint:          var_10 + z_k10_s430 + 1.17694 w_k10_HSBC_s43
                                 + 1.2618 w_k10_VOD_s43 + 1.12688 w_k10_BP_s43
                                 + 1.03283 w_k10_GSK_s43
                                 + 1.10442 w_k10_AZN_s43
                                 + 1.07762 w_k10_RIO_s43 + 1.01627 w_k10_BG_s43
                                 + 1.26739 w_k10_TSCO_s43
                                 + 1.03405 w_k10_AAL_s43
                                 + 1.06054 w_k10_PRU_s43 >= 1
 z_k10_s431_constraint:          var_10 + z_k10_s431 + 0.988969 w_k10_HSBC_s43
                                 + 0.97692 w_k10_VOD_s43
                                 + 0.843182 w_k10_BP_s43
                                 + 0.982495 w_k10_GSK_s43
                                 + 1.02852 w_k10_AZN_s43
                                 + 0.888451 w_k10_RIO_s43 + 1.0028 w_k10_BG_s43
                                 + 0.925558 w_k10_TSCO_s43
                                 + 1.00486 w_k10_AAL_s43
                                 + 0.926918 w_k10_PRU_s43 >= 1
 z_k10_s440_constraint:          var_10 + z_k10_s440 + 1.04236 w_k10_HSBC_s44
                                 + 1.03458 w_k10_VOD_s44 + 1.17434 w_k10_BP_s44
                                 + 1.00223 w_k10_GSK_s44
                                 + 0.939505 w_k10_AZN_s44
                                 + 1.0748 w_k10_RIO_s44 + 1.04136 w_k10_BG_s44
                                 + 1.06071 w_k10_TSCO_s44
                                 + 1.00086 w_k10_AAL_s44
                                 + 0.972754 w_k10_PRU_s44 >= 1
 z_k10_s441_constraint:          var_10 + z_k10_s441 + 0.899089 w_k10_HSBC_s44
                                 + 0.878443 w_k10_VOD_s44
                                 + 0.864929 w_k10_BP_s44
                                 + 0.839377 w_k10_GSK_s44
                                 + 0.893873 w_k10_AZN_s44
                                 + 0.826656 w_k10_RIO_s44
                                 + 0.970506 w_k10_BG_s44
                                 + 0.809895 w_k10_TSCO_s44
                                 + 0.982879 w_k10_AAL_s44
                                 + 0.988929 w_k10_PRU_s44 >= 1
 z_k10_s450_constraint:          var_10 + z_k10_s450 + 1.07487 w_k10_HSBC_s45
                                 + 0.98767 w_k10_VOD_s45
                                 + 0.920192 w_k10_BP_s45
                                 + 1.05267 w_k10_GSK_s45
                                 + 1.06121 w_k10_AZN_s45
                                 + 0.986212 w_k10_RIO_s45
                                 + 1.02318 w_k10_BG_s45
                                 + 1.04638 w_k10_TSCO_s45
                                 + 1.04742 w_k10_AAL_s45
                                 + 0.996725 w_k10_PRU_s45 >= 1
 z_k10_s451_constraint:          var_10 + z_k10_s451 + 1.13829 w_k10_HSBC_s45
                                 + 0.970328 w_k10_VOD_s45
                                 + 1.18995 w_k10_BP_s45 + 1.06374 w_k10_GSK_s45
                                 + 1.02795 w_k10_AZN_s45
                                 + 1.14419 w_k10_RIO_s45 + 1.0563 w_k10_BG_s45
                                 + 1.1643 w_k10_TSCO_s45
                                 + 1.09981 w_k10_AAL_s45
                                 + 1.08222 w_k10_PRU_s45 >= 1
 z_k10_s460_constraint:          var_10 + z_k10_s460 + 0.999867 w_k10_HSBC_s46
                                 + 0.994168 w_k10_VOD_s46
                                 + 0.901841 w_k10_BP_s46
                                 + 0.945658 w_k10_GSK_s46
                                 + 0.893881 w_k10_AZN_s46
                                 + 0.936741 w_k10_RIO_s46
                                 + 1.00178 w_k10_BG_s46
                                 + 0.860529 w_k10_TSCO_s46
                                 + 1.01302 w_k10_AAL_s46
                                 + 0.838406 w_k10_PRU_s46 >= 1
 z_k10_s461_constraint:          var_10 + z_k10_s461 + 0.994455 w_k10_HSBC_s46
                                 + 1.03822 w_k10_VOD_s46 + 1.26447 w_k10_BP_s46
                                 + 1.12276 w_k10_GSK_s46
                                 + 0.963287 w_k10_AZN_s46 + 1.1 w_k10_RIO_s46
                                 + 1.02823 w_k10_BG_s46
                                 + 1.16693 w_k10_TSCO_s46
                                 + 1.00458 w_k10_AAL_s46
                                 + 1.04907 w_k10_PRU_s46 >= 1
 z_k10_s470_constraint:          var_10 + z_k10_s470 + 1.00234 w_k10_HSBC_s47
                                 + 0.985365 w_k10_VOD_s47
                                 + 1.10938 w_k10_BP_s47 + 1.04114 w_k10_GSK_s47
                                 + 1.02743 w_k10_AZN_s47
                                 + 1.07434 w_k10_RIO_s47 + 1.02924 w_k10_BG_s47
                                 + 1.04586 w_k10_TSCO_s47
                                 + 1.09564 w_k10_AAL_s47
                                 + 0.998247 w_k10_PRU_s47 >= 1
 z_k10_s471_constraint:          var_10 + z_k10_s471 + 0.925746 w_k10_HSBC_s47
                                 + 1.07017 w_k10_VOD_s47
                                 + 0.878606 w_k10_BP_s47
                                 + 0.817843 w_k10_GSK_s47
                                 + 0.980701 w_k10_AZN_s47
                                 + 0.893799 w_k10_RIO_s47
                                 + 0.923361 w_k10_BG_s47
                                 + 0.879396 w_k10_TSCO_s47
                                 + 0.972719 w_k10_AAL_s47
                                 + 0.99052 w_k10_PRU_s47 >= 1
 z_k10_s480_constraint:          var_10 + z_k10_s480 + 1.02767 w_k10_HSBC_s48
                                 + 0.992538 w_k10_VOD_s48
                                 + 1.00789 w_k10_BP_s48 + 1.06395 w_k10_GSK_s48
                                 + 1.06635 w_k10_AZN_s48
                                 + 0.878547 w_k10_RIO_s48
                                 + 0.952378 w_k10_BG_s48
                                 + 0.90852 w_k10_TSCO_s48
                                 + 0.994644 w_k10_AAL_s48
                                 + 0.952919 w_k10_PRU_s48 >= 1
 z_k10_s481_constraint:          var_10 + z_k10_s481 + 1.08355 w_k10_HSBC_s48
                                 + 1.12084 w_k10_VOD_s48 + 1.23558 w_k10_BP_s48
                                 + 1.07849 w_k10_GSK_s48
                                 + 1.05856 w_k10_AZN_s48
                                 + 1.09662 w_k10_RIO_s48 + 1.10491 w_k10_BG_s48
                                 + 1.21964 w_k10_TSCO_s48
                                 + 1.09583 w_k10_AAL_s48
                                 + 1.13483 w_k10_PRU_s48 >= 1
 z_k10_s490_constraint:          var_10 + z_k10_s490 + 1.08398 w_k10_HSBC_s49
                                 + 0.989203 w_k10_VOD_s49
                                 + 0.980737 w_k10_BP_s49
                                 + 0.969939 w_k10_GSK_s49
                                 + 0.940458 w_k10_AZN_s49
                                 + 0.988807 w_k10_RIO_s49
                                 + 1.04088 w_k10_BG_s49
                                 + 0.869204 w_k10_TSCO_s49
                                 + 1.03664 w_k10_AAL_s49
                                 + 0.890497 w_k10_PRU_s49 >= 1
 z_k10_s491_constraint:          var_10 + z_k10_s491 + 1.1594 w_k10_HSBC_s49
                                 + 1.06158 w_k10_VOD_s49 + 1.1984 w_k10_BP_s49
                                 + 1.04989 w_k10_GSK_s49
                                 + 0.98681 w_k10_AZN_s49
                                 + 1.03316 w_k10_RIO_s49 + 1.15309 w_k10_BG_s49
                                 + 1.16601 w_k10_TSCO_s49
                                 + 1.08183 w_k10_AAL_s49
                                 + 0.959524 w_k10_PRU_s49 >= 1
 z_k10_s500_constraint:          var_10 + z_k10_s500 + 1.02963 w_k10_HSBC_s50
                                 + 0.972659 w_k10_VOD_s50
                                 + 1.20549 w_k10_BP_s50
                                 + 0.963374 w_k10_GSK_s50
                                 + 0.946899 w_k10_AZN_s50
                                 + 1.08258 w_k10_RIO_s50 + 1.02897 w_k10_BG_s50
                                 + 1.15986 w_k10_TSCO_s50
                                 + 1.02138 w_k10_AAL_s50
                                 + 0.952297 w_k10_PRU_s50 >= 1
 z_k10_s501_constraint:          var_10 + z_k10_s501 + 0.918961 w_k10_HSBC_s50
                                 + 1.09478 w_k10_VOD_s50
                                 + 0.866856 w_k10_BP_s50
                                 + 0.839643 w_k10_GSK_s50
                                 + 0.959179 w_k10_AZN_s50
                                 + 0.864423 w_k10_RIO_s50
                                 + 0.988605 w_k10_BG_s50
                                 + 0.82427 w_k10_TSCO_s50
                                 + 1.06542 w_k10_AAL_s50
                                 + 0.883427 w_k10_PRU_s50 >= 1
 z_k10_s510_constraint:          var_10 + z_k10_s510 + 0.933491 w_k10_HSBC_s51
                                 + 1.03427 w_k10_VOD_s51
                                 + 0.982067 w_k10_BP_s51
                                 + 0.95434 w_k10_GSK_s51
                                 + 1.00599 w_k10_AZN_s51
                                 + 0.965486 w_k10_RIO_s51
                                 + 0.966002 w_k10_BG_s51
                                 + 0.855053 w_k10_TSCO_s51
                                 + 0.877644 w_k10_AAL_s51
                                 + 0.918753 w_k10_PRU_s51 >= 1
 z_k10_s511_constraint:          var_10 + z_k10_s511 + 1.17745 w_k10_HSBC_s51
                                 + 1.15806 w_k10_VOD_s51 + 1.20261 w_k10_BP_s51
                                 + 1.01196 w_k10_GSK_s51
                                 + 1.04189 w_k10_AZN_s51
                                 + 1.12512 w_k10_RIO_s51 + 1.07218 w_k10_BG_s51
                                 + 1.24864 w_k10_TSCO_s51
                                 + 0.987072 w_k10_AAL_s51
                                 + 0.952499 w_k10_PRU_s51 >= 1
 z_k10_s520_constraint:          var_10 + z_k10_s520 + 1.06902 w_k10_HSBC_s52
                                 + 1.03033 w_k10_VOD_s52 + 1.1882 w_k10_BP_s52
                                 + 1.08222 w_k10_GSK_s52
                                 + 0.999228 w_k10_AZN_s52
                                 + 1.16215 w_k10_RIO_s52 + 1.04911 w_k10_BG_s52
                                 + 1.12114 w_k10_TSCO_s52
                                 + 1.0391 w_k10_AAL_s52 + 1.06639 w_k10_PRU_s52
                                 >= 1
 z_k10_s521_constraint:          var_10 + z_k10_s521 + 0.940672 w_k10_HSBC_s52
                                 + 0.940596 w_k10_VOD_s52
                                 + 0.807968 w_k10_BP_s52
                                 + 0.88492 w_k10_GSK_s52
                                 + 0.978863 w_k10_AZN_s52
                                 + 0.860868 w_k10_RIO_s52
                                 + 0.950439 w_k10_BG_s52
                                 + 0.851626 w_k10_TSCO_s52
                                 + 1.02024 w_k10_AAL_s52
                                 + 0.82689 w_k10_PRU_s52 >= 1
 z_k10_s530_constraint:          var_10 + z_k10_s530 + 1.00784 w_k10_HSBC_s53
                                 + 0.971574 w_k10_VOD_s53
                                 + 1.03808 w_k10_BP_s53 + 1.10368 w_k10_GSK_s53
                                 + 1.00624 w_k10_AZN_s53 + 1.1099 w_k10_RIO_s53
                                 + 1.02263 w_k10_BG_s53
                                 + 1.11957 w_k10_TSCO_s53
                                 + 1.00129 w_k10_AAL_s53 + 1.0217 w_k10_PRU_s53
                                 >= 1
 z_k10_s531_constraint:          var_10 + z_k10_s531 + 0.899786 w_k10_HSBC_s53
                                 + 1.00182 w_k10_VOD_s53 + 0.93844 w_k10_BP_s53
                                 + 0.941025 w_k10_GSK_s53
                                 + 0.979779 w_k10_AZN_s53
                                 + 0.879686 w_k10_RIO_s53
                                 + 0.886583 w_k10_BG_s53
                                 + 0.850978 w_k10_TSCO_s53
                                 + 0.943836 w_k10_AAL_s53
                                 + 0.936942 w_k10_PRU_s53 >= 1
 z_k10_s540_constraint:          var_10 + z_k10_s540 + 1.10152 w_k10_HSBC_s54
                                 + 0.898374 w_k10_VOD_s54
                                 + 1.21783 w_k10_BP_s54 + 1.15497 w_k10_GSK_s54
                                 + 1.05029 w_k10_AZN_s54
                                 + 1.07654 w_k10_RIO_s54 + 1.0667 w_k10_BG_s54
                                 + 1.1258 w_k10_TSCO_s54
                                 + 1.08295 w_k10_AAL_s54
                                 + 0.987677 w_k10_PRU_s54 >= 1
 z_k10_s541_constraint:          var_10 + z_k10_s541 + 0.924263 w_k10_HSBC_s54
                                 + 0.947941 w_k10_VOD_s54
                                 + 0.962547 w_k10_BP_s54
                                 + 1.05121 w_k10_GSK_s54
                                 + 0.983879 w_k10_AZN_s54
                                 + 1.01527 w_k10_RIO_s54 + 1.01647 w_k10_BG_s54
                                 + 0.902046 w_k10_TSCO_s54
                                 + 0.920219 w_k10_AAL_s54
                                 + 0.944492 w_k10_PRU_s54 >= 1
 z_k10_s550_constraint:          var_10 + z_k10_s550 + 1.03318 w_k10_HSBC_s55
                                 + 1.09216 w_k10_VOD_s55
                                 + 0.967347 w_k10_BP_s55
                                 + 1.03212 w_k10_GSK_s55 + 1.079 w_k10_AZN_s55
                                 + 1.00026 w_k10_RIO_s55
                                 + 0.958544 w_k10_BG_s55
                                 + 1.1668 w_k10_TSCO_s55
                                 + 0.93951 w_k10_AAL_s55
                                 + 0.981356 w_k10_PRU_s55 >= 1
 z_k10_s551_constraint:          var_10 + z_k10_s551 + 1.06032 w_k10_HSBC_s55
                                 + 1.07783 w_k10_VOD_s55 + 1.02273 w_k10_BP_s55
                                 + 1.00117 w_k10_GSK_s55
                                 + 0.981656 w_k10_AZN_s55
                                 + 1.13113 w_k10_RIO_s55 + 1.14354 w_k10_BG_s55
                                 + 0.990601 w_k10_TSCO_s55
                                 + 0.907197 w_k10_AAL_s55
                                 + 0.994654 w_k10_PRU_s55 >= 1
 z_k10_s560_constraint:          var_10 + z_k10_s560 + 1.00267 w_k10_HSBC_s56
                                 + 1.06712 w_k10_VOD_s56 + 1.0094 w_k10_BP_s56
                                 + 1.00109 w_k10_GSK_s56
                                 + 0.938614 w_k10_AZN_s56
                                 + 1.0185 w_k10_RIO_s56 + 1.08327 w_k10_BG_s56
                                 + 0.92855 w_k10_TSCO_s56
                                 + 1.00153 w_k10_AAL_s56
                                 + 0.925563 w_k10_PRU_s56 >= 1
 z_k10_s561_constraint:          var_10 + z_k10_s561 + 1.10408 w_k10_HSBC_s56
                                 + 1.05303 w_k10_VOD_s56 + 1.07835 w_k10_BP_s56
                                 + 1.09951 w_k10_GSK_s56
                                 + 1.12637 w_k10_AZN_s56
                                 + 1.09488 w_k10_RIO_s56 + 1.02331 w_k10_BG_s56
                                 + 1.11721 w_k10_TSCO_s56
                                 + 1.03097 w_k10_AAL_s56
                                 + 0.904562 w_k10_PRU_s56 >= 1
 z_k10_s570_constraint:          var_10 + z_k10_s570 + 1.0645 w_k10_HSBC_s57
                                 + 1.12407 w_k10_VOD_s57 + 1.18152 w_k10_BP_s57
                                 + 1.19718 w_k10_GSK_s57
                                 + 1.05231 w_k10_AZN_s57
                                 + 1.17446 w_k10_RIO_s57 + 1.08246 w_k10_BG_s57
                                 + 1.15229 w_k10_TSCO_s57
                                 + 0.998151 w_k10_AAL_s57
                                 + 1.14968 w_k10_PRU_s57 >= 1
 z_k10_s571_constraint:          var_10 + z_k10_s571 + 0.920764 w_k10_HSBC_s57
                                 + 0.934422 w_k10_VOD_s57
                                 + 0.913679 w_k10_BP_s57
                                 + 0.954671 w_k10_GSK_s57
                                 + 1.04124 w_k10_AZN_s57
                                 + 0.812569 w_k10_RIO_s57
                                 + 0.966582 w_k10_BG_s57
                                 + 0.874018 w_k10_TSCO_s57
                                 + 1.01562 w_k10_AAL_s57
                                 + 0.983157 w_k10_PRU_s57 >= 1
 z_k10_s580_constraint:          var_10 + z_k10_s580 + 1.02096 w_k10_HSBC_s58
                                 + 1.0448 w_k10_VOD_s58 + 0.9379 w_k10_BP_s58
                                 + 0.942824 w_k10_GSK_s58
                                 + 0.962746 w_k10_AZN_s58
                                 + 0.971547 w_k10_RIO_s58
                                 + 0.962613 w_k10_BG_s58
                                 + 0.824371 w_k10_TSCO_s58
                                 + 1.04222 w_k10_AAL_s58
                                 + 0.966126 w_k10_PRU_s58 >= 1
 z_k10_s581_constraint:          var_10 + z_k10_s581 + 1.07966 w_k10_HSBC_s58
                                 + 1.05953 w_k10_VOD_s58 + 1.14209 w_k10_BP_s58
                                 + 1.05834 w_k10_GSK_s58
                                 + 0.989096 w_k10_AZN_s58
                                 + 1.24474 w_k10_RIO_s58 + 1.05171 w_k10_BG_s58
                                 + 1.12565 w_k10_TSCO_s58
                                 + 0.90052 w_k10_AAL_s58
                                 + 1.08635 w_k10_PRU_s58 >= 1
 z_k10_s590_constraint:          var_10 + z_k10_s590 + 0.905688 w_k10_HSBC_s59
                                 + 0.969346 w_k10_VOD_s59
                                 + 0.912693 w_k10_BP_s59
                                 + 0.927017 w_k10_GSK_s59
                                 + 0.924619 w_k10_AZN_s59
                                 + 0.959425 w_k10_RIO_s59
                                 + 0.91432 w_k10_BG_s59
                                 + 0.931332 w_k10_TSCO_s59
                                 + 0.995296 w_k10_AAL_s59
                                 + 1.04851 w_k10_PRU_s59 >= 1
 z_k10_s591_constraint:          var_10 + z_k10_s591 + 1.01812 w_k10_HSBC_s59
                                 + 0.854578 w_k10_VOD_s59
                                 + 1.02113 w_k10_BP_s59 + 1.08345 w_k10_GSK_s59
                                 + 0.959656 w_k10_AZN_s59
                                 + 1.06164 w_k10_RIO_s59 + 1.04757 w_k10_BG_s59
                                 + 1.17372 w_k10_TSCO_s59
                                 + 0.963274 w_k10_AAL_s59
                                 + 0.970204 w_k10_PRU_s59 >= 1
 z_k10_s600_constraint:          var_10 + z_k10_s600 + 0.959656 w_k10_HSBC_s60
                                 + 1.12245 w_k10_VOD_s60
                                 + 0.902024 w_k10_BP_s60
                                 + 0.917963 w_k10_GSK_s60
                                 + 1.08379 w_k10_AZN_s60
                                 + 1.01346 w_k10_RIO_s60 + 1.02902 w_k10_BG_s60
                                 + 1.00883 w_k10_TSCO_s60
                                 + 1.06886 w_k10_AAL_s60
                                 + 1.09087 w_k10_PRU_s60 >= 1
 z_k10_s601_constraint:          var_10 + z_k10_s601 + 1.01013 w_k10_HSBC_s60
                                 + 0.945551 w_k10_VOD_s60
                                 + 1.02676 w_k10_BP_s60 + 1.05294 w_k10_GSK_s60
                                 + 0.962555 w_k10_AZN_s60
                                 + 1.05736 w_k10_RIO_s60 + 1.0954 w_k10_BG_s60
                                 + 0.985249 w_k10_TSCO_s60
                                 + 0.896867 w_k10_AAL_s60
                                 + 0.944453 w_k10_PRU_s60 >= 1
 z_k10_s610_constraint:          var_10 + z_k10_s610 + 1.14491 w_k10_HSBC_s61
                                 + 1.16715 w_k10_VOD_s61 + 1.19145 w_k10_BP_s61
                                 + 1.07046 w_k10_GSK_s61 + 1.0585 w_k10_AZN_s61
                                 + 1.1801 w_k10_RIO_s61 + 1.05047 w_k10_BG_s61
                                 + 1.08892 w_k10_TSCO_s61
                                 + 0.986026 w_k10_AAL_s61
                                 + 1.23587 w_k10_PRU_s61 >= 1
 z_k10_s611_constraint:          var_10 + z_k10_s611 + 1.05218 w_k10_HSBC_s61
                                 + 1.00337 w_k10_VOD_s61
                                 + 0.874339 w_k10_BP_s61
                                 + 0.883636 w_k10_GSK_s61
                                 + 0.997531 w_k10_AZN_s61
                                 + 0.83157 w_k10_RIO_s61
                                 + 0.835522 w_k10_BG_s61
                                 + 0.816514 w_k10_TSCO_s61
                                 + 0.979508 w_k10_AAL_s61
                                 + 1.19193 w_k10_PRU_s61 >= 1
 z_k10_s620_constraint:          var_10 + z_k10_s620 + 1.0114 w_k10_HSBC_s62
                                 + 1.05672 w_k10_VOD_s62 + 1.0472 w_k10_BP_s62
                                 + 0.983794 w_k10_GSK_s62
                                 + 1.02932 w_k10_AZN_s62
                                 + 1.05791 w_k10_RIO_s62 + 1.04551 w_k10_BG_s62
                                 + 1.01782 w_k10_TSCO_s62
                                 + 1.21229 w_k10_AAL_s62
                                 + 1.31931 w_k10_PRU_s62 >= 1
 z_k10_s621_constraint:          var_10 + z_k10_s621 + 1.00595 w_k10_HSBC_s62
                                 + 1.07014 w_k10_VOD_s62 + 1.09325 w_k10_BP_s62
                                 + 1.06069 w_k10_GSK_s62
                                 + 1.04369 w_k10_AZN_s62
                                 + 1.08967 w_k10_RIO_s62 + 1.02108 w_k10_BG_s62
                                 + 1.08445 w_k10_TSCO_s62
                                 + 0.988368 w_k10_AAL_s62
                                 + 1.04895 w_k10_PRU_s62 >= 1
 z_k10_s630_constraint:          var_10 + z_k10_s630 + 1.02742 w_k10_HSBC_s63
                                 + 1.04446 w_k10_VOD_s63
                                 + 0.923312 w_k10_BP_s63
                                 + 1.06858 w_k10_GSK_s63
                                 + 0.905483 w_k10_AZN_s63
                                 + 1.0271 w_k10_RIO_s63 + 0.987727 w_k10_BG_s63
                                 + 0.909853 w_k10_TSCO_s63
                                 + 0.86771 w_k10_AAL_s63
                                 + 1.11837 w_k10_PRU_s63 >= 1
 z_k10_s631_constraint:          var_10 + z_k10_s631 + 1.01472 w_k10_HSBC_s63
                                 + 1.00076 w_k10_VOD_s63 + 1.06708 w_k10_BP_s63
                                 + 0.978579 w_k10_GSK_s63
                                 + 0.965059 w_k10_AZN_s63
                                 + 1.04096 w_k10_RIO_s63 + 1.04012 w_k10_BG_s63
                                 + 0.989448 w_k10_TSCO_s63
                                 + 1.01226 w_k10_AAL_s63
                                 + 1.17397 w_k10_PRU_s63 >= 1
 z_k10_s640_constraint:          var_10 + z_k10_s640 + 1.06559 w_k10_HSBC_s64
                                 + 1.0449 w_k10_VOD_s64 + 0.900695 w_k10_BP_s64
                                 + 0.868765 w_k10_GSK_s64
                                 + 0.977255 w_k10_AZN_s64
                                 + 1.03665 w_k10_RIO_s64
                                 + 0.995907 w_k10_BG_s64
                                 + 0.776548 w_k10_TSCO_s64
                                 + 0.994209 w_k10_AAL_s64
                                 + 1.08168 w_k10_PRU_s64 >= 1
 z_k10_s641_constraint:          var_10 + z_k10_s641 + 1.12937 w_k10_HSBC_s64
                                 + 1.18515 w_k10_VOD_s64 + 1.13496 w_k10_BP_s64
                                 + 1.05442 w_k10_GSK_s64
                                 + 1.02095 w_k10_AZN_s64
                                 + 1.15699 w_k10_RIO_s64
                                 + 0.942208 w_k10_BG_s64
                                 + 1.06141 w_k10_TSCO_s64
                                 + 1.04833 w_k10_AAL_s64
                                 + 1.29422 w_k10_PRU_s64 >= 1
 z_k10_s650_constraint:          var_10 + z_k10_s650 + 0.781242 w_k10_HSBC_s65
                                 + 1.06605 w_k10_VOD_s65 + 0.96598 w_k10_BP_s65
                                 + 0.935417 w_k10_GSK_s65
                                 + 0.95285 w_k10_AZN_s65
                                 + 0.932422 w_k10_RIO_s65
                                 + 0.900052 w_k10_BG_s65
                                 + 0.932215 w_k10_TSCO_s65
                                 + 1.00765 w_k10_AAL_s65
                                 + 0.963617 w_k10_PRU_s65 >= 1
 z_k10_s651_constraint:          var_10 + z_k10_s651 + 0.957948 w_k10_HSBC_s65
                                 + 0.916147 w_k10_VOD_s65
                                 + 1.09579 w_k10_BP_s65 + 1.07884 w_k10_GSK_s65
                                 + 0.919103 w_k10_AZN_s65
                                 + 1.11292 w_k10_RIO_s65 + 1.02226 w_k10_BG_s65
                                 + 1.17284 w_k10_TSCO_s65
                                 + 0.961758 w_k10_AAL_s65
                                 + 1.11663 w_k10_PRU_s65 >= 1
 z_k10_s660_constraint:          var_10 + z_k10_s660 + 0.916148 w_k10_HSBC_s66
                                 + 0.926115 w_k10_VOD_s66
                                 + 0.936345 w_k10_BP_s66
                                 + 1.02644 w_k10_GSK_s66
                                 + 0.964012 w_k10_AZN_s66
                                 + 0.883168 w_k10_RIO_s66
                                 + 0.921917 w_k10_BG_s66
                                 + 1.05717 w_k10_TSCO_s66
                                 + 0.896831 w_k10_AAL_s66
                                 + 0.948746 w_k10_PRU_s66 >= 1
 z_k10_s661_constraint:          var_10 + z_k10_s661 + 1.1331 w_k10_HSBC_s66
                                 + 1.0022 w_k10_VOD_s66 + 1.03097 w_k10_BP_s66
                                 + 1.093 w_k10_GSK_s66 + 1.04096 w_k10_AZN_s66
                                 + 1.11529 w_k10_RIO_s66 + 1.07131 w_k10_BG_s66
                                 + 1.21873 w_k10_TSCO_s66
                                 + 0.942537 w_k10_AAL_s66
                                 + 1.09111 w_k10_PRU_s66 >= 1
 z_k10_s670_constraint:          var_10 + z_k10_s670 + 1.23853 w_k10_HSBC_s67
                                 + 1.06066 w_k10_VOD_s67 + 1.16427 w_k10_BP_s67
                                 + 1.06017 w_k10_GSK_s67
                                 + 1.05575 w_k10_AZN_s67
                                 + 1.24743 w_k10_RIO_s67 + 1.08045 w_k10_BG_s67
                                 + 1.20527 w_k10_TSCO_s67
                                 + 0.998569 w_k10_AAL_s67
                                 + 1.20114 w_k10_PRU_s67 >= 1
 z_k10_s671_constraint:          var_10 + z_k10_s671 + 1.02142 w_k10_HSBC_s67
                                 + 0.841516 w_k10_VOD_s67
                                 + 0.911467 w_k10_BP_s67
                                 + 0.989799 w_k10_GSK_s67
                                 + 0.924785 w_k10_AZN_s67
                                 + 0.907694 w_k10_RIO_s67
                                 + 1.06078 w_k10_BG_s67
                                 + 1.02812 w_k10_TSCO_s67
                                 + 0.978337 w_k10_AAL_s67
                                 + 0.878899 w_k10_PRU_s67 >= 1
 z_k10_s680_constraint:          var_10 + z_k10_s680 + 1.00558 w_k10_HSBC_s68
                                 + 0.879493 w_k10_VOD_s68
                                 + 0.989523 w_k10_BP_s68
                                 + 0.941519 w_k10_GSK_s68
                                 + 1.00522 w_k10_AZN_s68
                                 + 0.969927 w_k10_RIO_s68
                                 + 1.00184 w_k10_BG_s68
                                 + 1.01622 w_k10_TSCO_s68
                                 + 1.02487 w_k10_AAL_s68
                                 + 0.923689 w_k10_PRU_s68 >= 1
 z_k10_s681_constraint:          var_10 + z_k10_s681 + 1.14392 w_k10_HSBC_s68
                                 + 1.04628 w_k10_VOD_s68 + 1.12095 w_k10_BP_s68
                                 + 0.998009 w_k10_GSK_s68
                                 + 1.01911 w_k10_AZN_s68
                                 + 1.01949 w_k10_RIO_s68 + 1.03362 w_k10_BG_s68
                                 + 1.05629 w_k10_TSCO_s68
                                 + 1.11947 w_k10_AAL_s68
                                 + 0.888559 w_k10_PRU_s68 >= 1
 z_k10_s690_constraint:          var_10 + z_k10_s690 + 0.914114 w_k10_HSBC_s69
                                 + 0.953288 w_k10_VOD_s69
                                 + 0.864597 w_k10_BP_s69
                                 + 0.948339 w_k10_GSK_s69
                                 + 1.01655 w_k10_AZN_s69
                                 + 0.833042 w_k10_RIO_s69
                                 + 0.974008 w_k10_BG_s69
                                 + 0.949038 w_k10_TSCO_s69
                                 + 1.03497 w_k10_AAL_s69
                                 + 0.979064 w_k10_PRU_s69 >= 1
 z_k10_s691_constraint:          var_10 + z_k10_s691 + 1.13906 w_k10_HSBC_s69
                                 + 1.17788 w_k10_VOD_s69 + 1.0667 w_k10_BP_s69
                                 + 1.07626 w_k10_GSK_s69
                                 + 1.06801 w_k10_AZN_s69
                                 + 1.10956 w_k10_RIO_s69 + 1.04353 w_k10_BG_s69
                                 + 0.988185 w_k10_TSCO_s69
                                 + 0.972737 w_k10_AAL_s69
                                 + 1.07085 w_k10_PRU_s69 >= 1
 z_k10_s700_constraint:          var_10 + z_k10_s700 + 1.09901 w_k10_HSBC_s70
                                 + 0.862355 w_k10_VOD_s70
                                 + 0.941011 w_k10_BP_s70
                                 + 0.993121 w_k10_GSK_s70
                                 + 0.941482 w_k10_AZN_s70
                                 + 0.98414 w_k10_RIO_s70 + 1.04006 w_k10_BG_s70
                                 + 1.04317 w_k10_TSCO_s70
                                 + 0.924903 w_k10_AAL_s70
                                 + 0.840975 w_k10_PRU_s70 >= 1
 z_k10_s701_constraint:          var_10 + z_k10_s701 + 1.18957 w_k10_HSBC_s70
                                 + 1.1019 w_k10_VOD_s70 + 1.15739 w_k10_BP_s70
                                 + 1.04751 w_k10_GSK_s70
                                 + 1.08793 w_k10_AZN_s70
                                 + 1.04507 w_k10_RIO_s70
                                 + 0.946935 w_k10_BG_s70
                                 + 1.19241 w_k10_TSCO_s70
                                 + 1.03931 w_k10_AAL_s70 + 1.1169 w_k10_PRU_s70
                                 >= 1
 z_k10_s710_constraint:          var_10 + z_k10_s710 + 1.07348 w_k10_HSBC_s71
                                 + 0.914142 w_k10_VOD_s71
                                 + 1.23543 w_k10_BP_s71 + 1.25587 w_k10_GSK_s71
                                 + 1.09802 w_k10_AZN_s71
                                 + 1.14622 w_k10_RIO_s71 + 1.10011 w_k10_BG_s71
                                 + 1.44481 w_k10_TSCO_s71
                                 + 1.0536 w_k10_AAL_s71 + 1.2078 w_k10_PRU_s71
                                 >= 1
 z_k10_s711_constraint:          var_10 + z_k10_s711 + 0.945847 w_k10_HSBC_s71
                                 + 0.963206 w_k10_VOD_s71
                                 + 0.91337 w_k10_BP_s71 + 1.00717 w_k10_GSK_s71
                                 + 1.07906 w_k10_AZN_s71
                                 + 0.877661 w_k10_RIO_s71
                                 + 1.04819 w_k10_BG_s71
                                 + 0.958453 w_k10_TSCO_s71
                                 + 0.976494 w_k10_AAL_s71
                                 + 0.862187 w_k10_PRU_s71 >= 1
 z_k10_s720_constraint:          var_10 + z_k10_s720 + 0.876738 w_k10_HSBC_s72
                                 + 0.868463 w_k10_VOD_s72
                                 + 0.866803 w_k10_BP_s72
                                 + 0.924367 w_k10_GSK_s72
                                 + 0.95203 w_k10_AZN_s72
                                 + 0.762506 w_k10_RIO_s72
                                 + 0.871769 w_k10_BG_s72
                                 + 0.820288 w_k10_TSCO_s72
                                 + 0.973955 w_k10_AAL_s72
                                 + 0.886431 w_k10_PRU_s72 >= 1
 z_k10_s721_constraint:          var_10 + z_k10_s721 + 1.07881 w_k10_HSBC_s72
                                 + 0.966925 w_k10_VOD_s72
                                 + 1.08658 w_k10_BP_s72 + 1.12818 w_k10_GSK_s72
                                 + 0.976853 w_k10_AZN_s72
                                 + 1.00184 w_k10_RIO_s72
                                 + 0.972962 w_k10_BG_s72
                                 + 1.08582 w_k10_TSCO_s72
                                 + 0.978461 w_k10_AAL_s72
                                 + 0.991119 w_k10_PRU_s72 >= 1
 z_k10_s730_constraint:          var_10 + z_k10_s730 + 0.942262 w_k10_HSBC_s73
                                 + 1.02509 w_k10_VOD_s73 + 0.9362 w_k10_BP_s73
                                 + 0.945731 w_k10_GSK_s73
                                 + 1.04047 w_k10_AZN_s73
                                 + 1.03751 w_k10_RIO_s73
                                 + 0.980532 w_k10_BG_s73
                                 + 0.876352 w_k10_TSCO_s73
                                 + 1.01861 w_k10_AAL_s73
                                 + 1.05297 w_k10_PRU_s73 >= 1
 z_k10_s731_constraint:          var_10 + z_k10_s731 + 1.3089 w_k10_HSBC_s73
                                 + 1.04851 w_k10_VOD_s73 + 1.16282 w_k10_BP_s73
                                 + 0.973961 w_k10_GSK_s73
                                 + 0.938658 w_k10_AZN_s73
                                 + 1.26766 w_k10_RIO_s73 + 1.04584 w_k10_BG_s73
                                 + 1.07783 w_k10_TSCO_s73
                                 + 1.02561 w_k10_AAL_s73
                                 + 1.29007 w_k10_PRU_s73 >= 1
 z_k10_s740_constraint:          var_10 + z_k10_s740 + 1.1429 w_k10_HSBC_s74
                                 + 1.04531 w_k10_VOD_s74
                                 + 0.950209 w_k10_BP_s74
                                 + 0.975604 w_k10_GSK_s74
                                 + 1.0321 w_k10_AZN_s74
                                 + 0.964031 w_k10_RIO_s74
                                 + 0.908733 w_k10_BG_s74
                                 + 0.793425 w_k10_TSCO_s74
                                 + 1.02565 w_k10_AAL_s74
                                 + 0.956154 w_k10_PRU_s74 >= 1
 z_k10_s741_constraint:          var_10 + z_k10_s741 + 1.026 w_k10_HSBC_s74
                                 + 0.975936 w_k10_VOD_s74
                                 + 1.04098 w_k10_BP_s74
                                 + 0.985826 w_k10_GSK_s74
                                 + 0.971306 w_k10_AZN_s74
                                 + 1.1114 w_k10_RIO_s74 + 1.15235 w_k10_BG_s74
                                 + 1.0617 w_k10_TSCO_s74
                                 + 0.927342 w_k10_AAL_s74
                                 + 0.914941 w_k10_PRU_s74 >= 1
 z_k10_s750_constraint:          var_10 + z_k10_s750 + 0.921202 w_k10_HSBC_s75
                                 + 0.938895 w_k10_VOD_s75
                                 + 1.01939 w_k10_BP_s75
                                 + 0.975003 w_k10_GSK_s75
                                 + 1.01562 w_k10_AZN_s75
                                 + 1.01471 w_k10_RIO_s75
                                 + 0.978342 w_k10_BG_s75
                                 + 1.02163 w_k10_TSCO_s75
                                 + 1.02146 w_k10_AAL_s75
                                 + 0.97565 w_k10_PRU_s75 >= 1
 z_k10_s751_constraint:          var_10 + z_k10_s751 + 1.1044 w_k10_HSBC_s75
                                 + 0.947774 w_k10_VOD_s75
                                 + 0.958872 w_k10_BP_s75
                                 + 0.877536 w_k10_GSK_s75
                                 + 0.862685 w_k10_AZN_s75
                                 + 1.10913 w_k10_RIO_s75 + 1.06968 w_k10_BG_s75
                                 + 0.924351 w_k10_TSCO_s75
                                 + 0.883432 w_k10_AAL_s75
                                 + 1.02047 w_k10_PRU_s75 >= 1
 z_k10_s760_constraint:          var_10 + z_k10_s760 + 1.1807 w_k10_HSBC_s76
                                 + 1.02909 w_k10_VOD_s76 + 1.11505 w_k10_BP_s76
                                 + 1.10378 w_k10_GSK_s76
                                 + 1.11857 w_k10_AZN_s76
                                 + 1.00389 w_k10_RIO_s76 + 1.0076 w_k10_BG_s76
                                 + 1.27473 w_k10_TSCO_s76
                                 + 0.986891 w_k10_AAL_s76
                                 + 0.977927 w_k10_PRU_s76 >= 1
 z_k10_s761_constraint:          var_10 + z_k10_s761 + 0.972335 w_k10_HSBC_s76
                                 + 0.947983 w_k10_VOD_s76
                                 + 0.847667 w_k10_BP_s76 + 1.0098 w_k10_GSK_s76
                                 + 1.04977 w_k10_AZN_s76
                                 + 0.863585 w_k10_RIO_s76
                                 + 1.00541 w_k10_BG_s76
                                 + 0.964615 w_k10_TSCO_s76
                                 + 0.966683 w_k10_AAL_s76
                                 + 0.924937 w_k10_PRU_s76 >= 1
 z_k10_s770_constraint:          var_10 + z_k10_s770 + 0.824387 w_k10_HSBC_s77
                                 + 1.00073 w_k10_VOD_s77
                                 + 0.840515 w_k10_BP_s77
                                 + 0.847152 w_k10_GSK_s77
                                 + 0.904713 w_k10_AZN_s77
                                 + 0.895099 w_k10_RIO_s77
                                 + 0.937892 w_k10_BG_s77
                                 + 0.824964 w_k10_TSCO_s77
                                 + 0.999418 w_k10_AAL_s77
                                 + 0.871726 w_k10_PRU_s77 >= 1
 z_k10_s771_constraint:          var_10 + z_k10_s771 + 1.0294 w_k10_HSBC_s77
                                 + 0.893713 w_k10_VOD_s77
                                 + 1.12008 w_k10_BP_s77 + 1.06676 w_k10_GSK_s77
                                 + 0.932246 w_k10_AZN_s77
                                 + 1.03339 w_k10_RIO_s77 + 1.02987 w_k10_BG_s77
                                 + 1.10501 w_k10_TSCO_s77
                                 + 0.990327 w_k10_AAL_s77
                                 + 0.91539 w_k10_PRU_s77 >= 1
 z_k10_s780_constraint:          var_10 + z_k10_s780 + 0.946994 w_k10_HSBC_s78
                                 + 1.06255 w_k10_VOD_s78 + 1.09438 w_k10_BP_s78
                                 + 1.05496 w_k10_GSK_s78
                                 + 1.01136 w_k10_AZN_s78
                                 + 1.11524 w_k10_RIO_s78
                                 + 0.946454 w_k10_BG_s78
                                 + 1.02942 w_k10_TSCO_s78
                                 + 1.07895 w_k10_AAL_s78
                                 + 1.20957 w_k10_PRU_s78 >= 1
 z_k10_s781_constraint:          var_10 + z_k10_s781 + 1.00843 w_k10_HSBC_s78
                                 + 0.947864 w_k10_VOD_s78
                                 + 0.97112 w_k10_BP_s78
                                 + 0.939056 w_k10_GSK_s78
                                 + 1.00267 w_k10_AZN_s78
                                 + 1.11668 w_k10_RIO_s78 + 1.03085 w_k10_BG_s78
                                 + 1.11765 w_k10_TSCO_s78
                                 + 0.953133 w_k10_AAL_s78
                                 + 0.963779 w_k10_PRU_s78 >= 1
 z_k10_s790_constraint:          var_10 + z_k10_s790 + 1.10165 w_k10_HSBC_s79
                                 + 1.15338 w_k10_VOD_s79 + 1.20433 w_k10_BP_s79
                                 + 1.05372 w_k10_GSK_s79
                                 + 1.12877 w_k10_AZN_s79
                                 + 1.07421 w_k10_RIO_s79 + 1.04048 w_k10_BG_s79
                                 + 1.2537 w_k10_TSCO_s79
                                 + 1.05498 w_k10_AAL_s79
                                 + 1.24723 w_k10_PRU_s79 >= 1
 z_k10_s791_constraint:          var_10 + z_k10_s791 + 0.895471 w_k10_HSBC_s79
                                 + 0.948458 w_k10_VOD_s79
                                 + 0.884353 w_k10_BP_s79
                                 + 0.928465 w_k10_GSK_s79
                                 + 1.03837 w_k10_AZN_s79
                                 + 0.887033 w_k10_RIO_s79
                                 + 0.975966 w_k10_BG_s79
                                 + 0.796989 w_k10_TSCO_s79
                                 + 1.03275 w_k10_AAL_s79 + 1.1413 w_k10_PRU_s79
                                 >= 1
 z_k10_s800_constraint:          var_10 + z_k10_s800 + 1.10301 w_k10_HSBC_s80
                                 + 1.01621 w_k10_VOD_s80 + 1.14113 w_k10_BP_s80
                                 + 1.05078 w_k10_GSK_s80
                                 + 1.00827 w_k10_AZN_s80
                                 + 1.11558 w_k10_RIO_s80
                                 + 0.994224 w_k10_BG_s80
                                 + 1.09121 w_k10_TSCO_s80
                                 + 0.951409 w_k10_AAL_s80
                                 + 1.04774 w_k10_PRU_s80 >= 1
 z_k10_s801_constraint:          var_10 + z_k10_s801 + 0.958903 w_k10_HSBC_s80
                                 + 0.958348 w_k10_VOD_s80
                                 + 0.803981 w_k10_BP_s80
                                 + 0.988311 w_k10_GSK_s80
                                 + 1.00291 w_k10_AZN_s80
                                 + 0.857175 w_k10_RIO_s80
                                 + 0.994608 w_k10_BG_s80
                                 + 0.783354 w_k10_TSCO_s80
                                 + 0.891694 w_k10_AAL_s80
                                 + 0.748313 w_k10_PRU_s80 >= 1
 z_k10_s810_constraint:          var_10 + z_k10_s810 + 0.936235 w_k10_HSBC_s81
                                 + 0.99824 w_k10_VOD_s81
                                 + 0.933344 w_k10_BP_s81
                                 + 1.03304 w_k10_GSK_s81
                                 + 0.969669 w_k10_AZN_s81
                                 + 0.978054 w_k10_RIO_s81
                                 + 1.02403 w_k10_BG_s81
                                 + 0.941464 w_k10_TSCO_s81
                                 + 0.943134 w_k10_AAL_s81
                                 + 0.8925 w_k10_PRU_s81 >= 1
 z_k10_s811_constraint:          var_10 + z_k10_s811 + 1.0439 w_k10_HSBC_s81
                                 + 1.08379 w_k10_VOD_s81 + 1.07489 w_k10_BP_s81
                                 + 1.16159 w_k10_GSK_s81
                                 + 1.06195 w_k10_AZN_s81
                                 + 1.02221 w_k10_RIO_s81 + 1.00203 w_k10_BG_s81
                                 + 1.16879 w_k10_TSCO_s81
                                 + 1.15876 w_k10_AAL_s81
                                 + 1.04161 w_k10_PRU_s81 >= 1
 z_k10_s820_constraint:          var_10 + z_k10_s820 + 1.02006 w_k10_HSBC_s82
                                 + 0.923241 w_k10_VOD_s82
                                 + 0.948491 w_k10_BP_s82
                                 + 1.02958 w_k10_GSK_s82
                                 + 0.965854 w_k10_AZN_s82
                                 + 0.976823 w_k10_RIO_s82
                                 + 0.976093 w_k10_BG_s82
                                 + 0.856747 w_k10_TSCO_s82
                                 + 1.06396 w_k10_AAL_s82
                                 + 0.929984 w_k10_PRU_s82 >= 1
 z_k10_s821_constraint:          var_10 + z_k10_s821 + 1.1236 w_k10_HSBC_s82
                                 + 0.950829 w_k10_VOD_s82
                                 + 1.26354 w_k10_BP_s82 + 1.20215 w_k10_GSK_s82
                                 + 1.02244 w_k10_AZN_s82
                                 + 1.21813 w_k10_RIO_s82 + 1.0365 w_k10_BG_s82
                                 + 1.26978 w_k10_TSCO_s82
                                 + 1.06776 w_k10_AAL_s82
                                 + 1.15358 w_k10_PRU_s82 >= 1
 z_k10_s830_constraint:          var_10 + z_k10_s830 + 1.08487 w_k10_HSBC_s83
                                 + 1.14142 w_k10_VOD_s83 + 1.16767 w_k10_BP_s83
                                 + 0.948405 w_k10_GSK_s83
                                 + 0.905285 w_k10_AZN_s83
                                 + 1.16461 w_k10_RIO_s83 + 1.10967 w_k10_BG_s83
                                 + 1.31996 w_k10_TSCO_s83
                                 + 0.961582 w_k10_AAL_s83
                                 + 0.99181 w_k10_PRU_s83 >= 1
 z_k10_s831_constraint:          var_10 + z_k10_s831 + 1.00515 w_k10_HSBC_s83
                                 + 1.15735 w_k10_VOD_s83 + 0.92153 w_k10_BP_s83
                                 + 0.879603 w_k10_GSK_s83
                                 + 1.03552 w_k10_AZN_s83
                                 + 0.927495 w_k10_RIO_s83
                                 + 0.981877 w_k10_BG_s83
                                 + 0.997706 w_k10_TSCO_s83
                                 + 1.02563 w_k10_AAL_s83
                                 + 1.01314 w_k10_PRU_s83 >= 1
 z_k10_s840_constraint:          var_10 + z_k10_s840 + 1.02828 w_k10_HSBC_s84
                                 + 0.975542 w_k10_VOD_s84
                                 + 1.03244 w_k10_BP_s84
                                 + 0.996569 w_k10_GSK_s84
                                 + 0.973688 w_k10_AZN_s84
                                 + 0.968787 w_k10_RIO_s84
                                 + 0.973743 w_k10_BG_s84
                                 + 0.963252 w_k10_TSCO_s84
                                 + 1.01727 w_k10_AAL_s84
                                 + 1.00885 w_k10_PRU_s84 >= 1
 z_k10_s841_constraint:          var_10 + z_k10_s841 + 0.895659 w_k10_HSBC_s84
                                 + 0.982295 w_k10_VOD_s84
                                 + 1.12445 w_k10_BP_s84 + 1.10952 w_k10_GSK_s84
                                 + 0.91746 w_k10_AZN_s84
                                 + 1.03262 w_k10_RIO_s84 + 1.12685 w_k10_BG_s84
                                 + 1.0301 w_k10_TSCO_s84
                                 + 0.990394 w_k10_AAL_s84
                                 + 1.02629 w_k10_PRU_s84 >= 1
 z_k10_s850_constraint:          var_10 + z_k10_s850 + 1.05383 w_k10_HSBC_s85
                                 + 1.14838 w_k10_VOD_s85 + 1.12976 w_k10_BP_s85
                                 + 1.03968 w_k10_GSK_s85
                                 + 1.12765 w_k10_AZN_s85 + 1.0991 w_k10_RIO_s85
                                 + 1.04179 w_k10_BG_s85
                                 + 1.09826 w_k10_TSCO_s85
                                 + 1.05487 w_k10_AAL_s85
                                 + 1.10603 w_k10_PRU_s85 >= 1
 z_k10_s851_constraint:          var_10 + z_k10_s851 + 0.968482 w_k10_HSBC_s85
                                 + 0.970865 w_k10_VOD_s85
                                 + 0.892698 w_k10_BP_s85
                                 + 0.943943 w_k10_GSK_s85
                                 + 1.02943 w_k10_AZN_s85
                                 + 0.889535 w_k10_RIO_s85
                                 + 0.934956 w_k10_BG_s85
                                 + 0.818167 w_k10_TSCO_s85
                                 + 1.03766 w_k10_AAL_s85
                                 + 0.945455 w_k10_PRU_s85 >= 1
 z_k10_s860_constraint:          var_10 + z_k10_s860 + 0.967345 w_k10_HSBC_s86
                                 + 0.967558 w_k10_VOD_s86
                                 + 0.970886 w_k10_BP_s86
                                 + 1.00223 w_k10_GSK_s86
                                 + 0.981571 w_k10_AZN_s86
                                 + 1.01622 w_k10_RIO_s86
                                 + 0.955357 w_k10_BG_s86
                                 + 1.06691 w_k10_TSCO_s86
                                 + 0.947428 w_k10_AAL_s86
                                 + 1.06781 w_k10_PRU_s86 >= 1
 z_k10_s861_constraint:          var_10 + z_k10_s861 + 1.1951 w_k10_HSBC_s86
                                 + 1.07675 w_k10_VOD_s86 + 1.16939 w_k10_BP_s86
                                 + 1.05725 w_k10_GSK_s86
                                 + 1.04288 w_k10_AZN_s86
                                 + 1.08693 w_k10_RIO_s86
                                 + 0.986744 w_k10_BG_s86
                                 + 1.19531 w_k10_TSCO_s86
                                 + 1.17921 w_k10_AAL_s86
                                 + 1.25862 w_k10_PRU_s86 >= 1
 z_k10_s870_constraint:          var_10 + z_k10_s870 + 0.98098 w_k10_HSBC_s87
                                 + 1.05001 w_k10_VOD_s87
                                 + 0.965326 w_k10_BP_s87
                                 + 0.951358 w_k10_GSK_s87
                                 + 1.10224 w_k10_AZN_s87
                                 + 0.894385 w_k10_RIO_s87
                                 + 1.02377 w_k10_BG_s87
                                 + 0.905387 w_k10_TSCO_s87
                                 + 0.996505 w_k10_AAL_s87
                                 + 0.965562 w_k10_PRU_s87 >= 1
 z_k10_s871_constraint:          var_10 + z_k10_s871 + 1.10372 w_k10_HSBC_s87
                                 + 1.07913 w_k10_VOD_s87 + 1.17554 w_k10_BP_s87
                                 + 1.13721 w_k10_GSK_s87
                                 + 1.08944 w_k10_AZN_s87
                                 + 1.09419 w_k10_RIO_s87 + 1.01948 w_k10_BG_s87
                                 + 1.17402 w_k10_TSCO_s87
                                 + 1.11397 w_k10_AAL_s87
                                 + 1.02238 w_k10_PRU_s87 >= 1
 z_k10_s880_constraint:          var_10 + z_k10_s880 + 1.11456 w_k10_HSBC_s88
                                 + 1.09513 w_k10_VOD_s88 + 1.01402 w_k10_BP_s88
                                 + 1.00941 w_k10_GSK_s88
                                 + 1.00844 w_k10_AZN_s88 + 1.0526 w_k10_RIO_s88
                                 + 0.987552 w_k10_BG_s88
                                 + 1.06909 w_k10_TSCO_s88
                                 + 0.93474 w_k10_AAL_s88
                                 + 1.11592 w_k10_PRU_s88 >= 1
 z_k10_s881_constraint:          var_10 + z_k10_s881 + 0.916706 w_k10_HSBC_s88
                                 + 0.921962 w_k10_VOD_s88
                                 + 1.01047 w_k10_BP_s88 + 1.10495 w_k10_GSK_s88
                                 + 1.03692 w_k10_AZN_s88
                                 + 0.889416 w_k10_RIO_s88
                                 + 0.946906 w_k10_BG_s88
                                 + 1.09162 w_k10_TSCO_s88
                                 + 1.12825 w_k10_AAL_s88
                                 + 0.934138 w_k10_PRU_s88 >= 1
 z_k10_s890_constraint:          var_10 + z_k10_s890 + 1.00764 w_k10_HSBC_s89
                                 + 0.829581 w_k10_VOD_s89
                                 + 0.999401 w_k10_BP_s89
                                 + 0.987722 w_k10_GSK_s89
                                 + 0.973677 w_k10_AZN_s89
                                 + 0.949021 w_k10_RIO_s89
                                 + 0.967127 w_k10_BG_s89
                                 + 0.998337 w_k10_TSCO_s89
                                 + 1.04614 w_k10_AAL_s89
                                 + 0.862998 w_k10_PRU_s89 >= 1
 z_k10_s891_constraint:          var_10 + z_k10_s891 + 1.26351 w_k10_HSBC_s89
                                 + 1.1188 w_k10_VOD_s89 + 1.3714 w_k10_BP_s89
                                 + 1.16829 w_k10_GSK_s89
                                 + 1.02362 w_k10_AZN_s89
                                 + 1.38408 w_k10_RIO_s89 + 1.07355 w_k10_BG_s89
                                 + 1.3989 w_k10_TSCO_s89
                                 + 0.96307 w_k10_AAL_s89
                                 + 1.05958 w_k10_PRU_s89 >= 1
 z_k10_s900_constraint:          var_10 + z_k10_s900 + 1.13688 w_k10_HSBC_s90
                                 + 0.963054 w_k10_VOD_s90
                                 + 1.19033 w_k10_BP_s90 + 1.11702 w_k10_GSK_s90
                                 + 1.03114 w_k10_AZN_s90
                                 + 1.07813 w_k10_RIO_s90 + 1.07274 w_k10_BG_s90
                                 + 1.17678 w_k10_TSCO_s90
                                 + 1.08231 w_k10_AAL_s90
                                 + 0.822338 w_k10_PRU_s90 >= 1
 z_k10_s901_constraint:          var_10 + z_k10_s901 + 0.994443 w_k10_HSBC_s90
                                 + 0.94258 w_k10_VOD_s90
                                 + 0.908972 w_k10_BP_s90
                                 + 1.01694 w_k10_GSK_s90
                                 + 1.10438 w_k10_AZN_s90
                                 + 0.888162 w_k10_RIO_s90
                                 + 0.979305 w_k10_BG_s90
                                 + 0.991957 w_k10_TSCO_s90
                                 + 1.02057 w_k10_AAL_s90
                                 + 0.828288 w_k10_PRU_s90 >= 1
 z_k10_s910_constraint:          var_10 + z_k10_s910 + 1.19437 w_k10_HSBC_s91
                                 + 1.13657 w_k10_VOD_s91 + 1.2686 w_k10_BP_s91
                                 + 1.19361 w_k10_GSK_s91
                                 + 1.03808 w_k10_AZN_s91
                                 + 1.25828 w_k10_RIO_s91 + 1.07417 w_k10_BG_s91
                                 + 1.0938 w_k10_TSCO_s91
                                 + 1.02818 w_k10_AAL_s91
                                 + 1.08183 w_k10_PRU_s91 >= 1
 z_k10_s911_constraint:          var_10 + z_k10_s911 + 1.11236 w_k10_HSBC_s91
                                 + 0.932742 w_k10_VOD_s91
                                 + 0.917074 w_k10_BP_s91
                                 + 1.08286 w_k10_GSK_s91
                                 + 1.02112 w_k10_AZN_s91
                                 + 1.01105 w_k10_RIO_s91 + 1.00209 w_k10_BG_s91
                                 + 0.963191 w_k10_TSCO_s91
                                 + 0.945705 w_k10_AAL_s91
                                 + 0.984855 w_k10_PRU_s91 >= 1
 z_k10_s920_constraint:          var_10 + z_k10_s920 + 0.959196 w_k10_HSBC_s92
                                 + 1.03064 w_k10_VOD_s92
                                 + 0.914097 w_k10_BP_s92
                                 + 0.856141 w_k10_GSK_s92
                                 + 0.997587 w_k10_AZN_s92
                                 + 0.991796 w_k10_RIO_s92
                                 + 0.987001 w_k10_BG_s92
                                 + 0.82733 w_k10_TSCO_s92
                                 + 0.885138 w_k10_AAL_s92
                                 + 0.936005 w_k10_PRU_s92 >= 1
 z_k10_s921_constraint:          var_10 + z_k10_s921 + 1.17439 w_k10_HSBC_s92
                                 + 1.08037 w_k10_VOD_s92 + 1.25502 w_k10_BP_s92
                                 + 1.00962 w_k10_GSK_s92
                                 + 0.969939 w_k10_AZN_s92
                                 + 1.27295 w_k10_RIO_s92 + 1.12396 w_k10_BG_s92
                                 + 1.15235 w_k10_TSCO_s92
                                 + 0.926834 w_k10_AAL_s92
                                 + 1.18892 w_k10_PRU_s92 >= 1
 z_k10_s930_constraint:          var_10 + z_k10_s930 + 1.07003 w_k10_HSBC_s93
                                 + 1.01492 w_k10_VOD_s93 + 1.14894 w_k10_BP_s93
                                 + 1.04881 w_k10_GSK_s93
                                 + 1.04512 w_k10_AZN_s93
                                 + 0.984372 w_k10_RIO_s93
                                 + 1.05521 w_k10_BG_s93
                                 + 1.10877 w_k10_TSCO_s93
                                 + 0.938559 w_k10_AAL_s93
                                 + 1.00419 w_k10_PRU_s93 >= 1
 z_k10_s931_constraint:          var_10 + z_k10_s931 + 1.01511 w_k10_HSBC_s93
                                 + 0.898611 w_k10_VOD_s93
                                 + 0.939895 w_k10_BP_s93
                                 + 0.944754 w_k10_GSK_s93
                                 + 0.983857 w_k10_AZN_s93
                                 + 0.965218 w_k10_RIO_s93
                                 + 1.00469 w_k10_BG_s93
                                 + 0.906274 w_k10_TSCO_s93
                                 + 0.947681 w_k10_AAL_s93
                                 + 0.897636 w_k10_PRU_s93 >= 1
 z_k10_s940_constraint:          var_10 + z_k10_s940 + 0.953346 w_k10_HSBC_s94
                                 + 0.990632 w_k10_VOD_s94 + 1.1006 w_k10_BP_s94
                                 + 1.09759 w_k10_GSK_s94
                                 + 1.09358 w_k10_AZN_s94 + 1.0023 w_k10_RIO_s94
                                 + 1.03143 w_k10_BG_s94
                                 + 1.21342 w_k10_TSCO_s94
                                 + 0.982482 w_k10_AAL_s94
                                 + 0.949189 w_k10_PRU_s94 >= 1
 z_k10_s941_constraint:          var_10 + z_k10_s941 + 0.812245 w_k10_HSBC_s94
                                 + 0.885568 w_k10_VOD_s94
                                 + 0.800554 w_k10_BP_s94
                                 + 0.961638 w_k10_GSK_s94
                                 + 0.927245 w_k10_AZN_s94
                                 + 0.762032 w_k10_RIO_s94
                                 + 0.977295 w_k10_BG_s94
                                 + 0.881483 w_k10_TSCO_s94
                                 + 1.14447 w_k10_AAL_s94
                                 + 0.928395 w_k10_PRU_s94 >= 1
 z_k10_s950_constraint:          var_10 + z_k10_s950 + 1.08349 w_k10_HSBC_s95
                                 + 0.862779 w_k10_VOD_s95
                                 + 1.00999 w_k10_BP_s95 + 1.01042 w_k10_GSK_s95
                                 + 0.960401 w_k10_AZN_s95
                                 + 0.943829 w_k10_RIO_s95
                                 + 0.954813 w_k10_BG_s95
                                 + 0.854948 w_k10_TSCO_s95
                                 + 0.962875 w_k10_AAL_s95
                                 + 1.04629 w_k10_PRU_s95 >= 1
 z_k10_s951_constraint:          var_10 + z_k10_s951 + 0.975546 w_k10_HSBC_s95
                                 + 0.914487 w_k10_VOD_s95
                                 + 1.20025 w_k10_BP_s95 + 1.09888 w_k10_GSK_s95
                                 + 1.08351 w_k10_AZN_s95
                                 + 0.971466 w_k10_RIO_s95
                                 + 1.00666 w_k10_BG_s95
                                 + 1.12458 w_k10_TSCO_s95
                                 + 1.12156 w_k10_AAL_s95
                                 + 1.13021 w_k10_PRU_s95 >= 1
 z_k10_s960_constraint:          var_10 + z_k10_s960 + 1.00257 w_k10_HSBC_s96
                                 + 0.964574 w_k10_VOD_s96
                                 + 1.01913 w_k10_BP_s96 + 1.10257 w_k10_GSK_s96
                                 + 1.04521 w_k10_AZN_s96 + 1.1439 w_k10_RIO_s96
                                 + 1.04677 w_k10_BG_s96
                                 + 1.09688 w_k10_TSCO_s96
                                 + 0.951074 w_k10_AAL_s96
                                 + 0.986779 w_k10_PRU_s96 >= 1
 z_k10_s961_constraint:          var_10 + z_k10_s961 + 0.934182 w_k10_HSBC_s96
                                 + 0.981219 w_k10_VOD_s96 + 0.9417 w_k10_BP_s96
                                 + 0.955875 w_k10_GSK_s96
                                 + 1.07947 w_k10_AZN_s96
                                 + 0.86197 w_k10_RIO_s96
                                 + 0.928992 w_k10_BG_s96
                                 + 0.848607 w_k10_TSCO_s96
                                 + 1.12642 w_k10_AAL_s96
                                 + 1.06191 w_k10_PRU_s96 >= 1
 z_k10_s970_constraint:          var_10 + z_k10_s970 + 1.08457 w_k10_HSBC_s97
                                 + 1.07329 w_k10_VOD_s97 + 1.08148 w_k10_BP_s97
                                 + 1.02639 w_k10_GSK_s97 + 1.058 w_k10_AZN_s97
                                 + 1.10862 w_k10_RIO_s97 + 1.01415 w_k10_BG_s97
                                 + 1.03058 w_k10_TSCO_s97
                                 + 1.00778 w_k10_AAL_s97
                                 + 1.20167 w_k10_PRU_s97 >= 1
 z_k10_s971_constraint:          var_10 + z_k10_s971 + 0.884818 w_k10_HSBC_s97
                                 + 0.837945 w_k10_VOD_s97
                                 + 0.877065 w_k10_BP_s97
                                 + 0.982598 w_k10_GSK_s97
                                 + 1.06069 w_k10_AZN_s97
                                 + 0.872103 w_k10_RIO_s97
                                 + 0.93583 w_k10_BG_s97
                                 + 0.880577 w_k10_TSCO_s97
                                 + 0.998228 w_k10_AAL_s97
                                 + 0.98764 w_k10_PRU_s97 >= 1
 z_k10_s980_constraint:          var_10 + z_k10_s980 + 1.25861 w_k10_HSBC_s98
                                 + 1.16128 w_k10_VOD_s98 + 1.17627 w_k10_BP_s98
                                 + 1.09308 w_k10_GSK_s98
                                 + 1.06465 w_k10_AZN_s98
                                 + 1.10544 w_k10_RIO_s98 + 1.02442 w_k10_BG_s98
                                 + 1.15667 w_k10_TSCO_s98
                                 + 1.00071 w_k10_AAL_s98
                                 + 1.02782 w_k10_PRU_s98 >= 1
 z_k10_s981_constraint:          var_10 + z_k10_s981 + 0.996998 w_k10_HSBC_s98
                                 + 0.990808 w_k10_VOD_s98
                                 + 0.818385 w_k10_BP_s98
                                 + 1.01755 w_k10_GSK_s98
                                 + 0.973407 w_k10_AZN_s98
                                 + 0.861731 w_k10_RIO_s98
                                 + 0.994587 w_k10_BG_s98
                                 + 0.955471 w_k10_TSCO_s98
                                 + 0.924841 w_k10_AAL_s98
                                 + 0.880864 w_k10_PRU_s98 >= 1
 z_k10_s990_constraint:          var_10 + z_k10_s990 + 0.96245 w_k10_HSBC_s99
                                 + 1.01709 w_k10_VOD_s99
                                 + 0.947782 w_k10_BP_s99
                                 + 0.996951 w_k10_GSK_s99
                                 + 1.02284 w_k10_AZN_s99
                                 + 0.961171 w_k10_RIO_s99
                                 + 0.996306 w_k10_BG_s99
                                 + 0.917151 w_k10_TSCO_s99
                                 + 1.02344 w_k10_AAL_s99
                                 + 1.02409 w_k10_PRU_s99 >= 1
 z_k10_s991_constraint:          var_10 + z_k10_s991 + 1.19584 w_k10_HSBC_s99
                                 + 1.07875 w_k10_VOD_s99 + 1.22927 w_k10_BP_s99
                                 + 1.20151 w_k10_GSK_s99
                                 + 1.04648 w_k10_AZN_s99
                                 + 1.30382 w_k10_RIO_s99 + 1.09787 w_k10_BG_s99
                                 + 1.07313 w_k10_TSCO_s99
                                 + 1.02182 w_k10_AAL_s99
                                 + 1.03225 w_k10_PRU_s99 >= 1
 k10_return_constraint:          0.0346568187827807 w_k10_HSBC_s00
                                 + 0.0109229705648393 w_k10_HSBC_s01
                                 + 0.0104970217632178 w_k10_HSBC_s02
                                 + 0.00966250156427317 w_k10_HSBC_s03
                                 + 0.00735933812903 w_k10_HSBC_s04
                                 + 0.0130551713136074 w_k10_HSBC_s05
                                 + 0.0150394366030544 w_k10_HSBC_s06
                                 + 0.00831876837938647 w_k10_HSBC_s07
                                 + 0.00689188560767393 w_k10_HSBC_s08
                                 + 0.0126909723453042 w_k10_HSBC_s09
                                 + 0.0115906616466421 w_k10_HSBC_s10
                                 + 0.00725458658866052 w_k10_HSBC_s11
                                 + 0.0152923049390932 w_k10_HSBC_s12
                                 + 0.00645561110744737 w_k10_HSBC_s13
                                 + 0.0174380494969452 w_k10_HSBC_s14
                                 + 0.00670681885204998 w_k10_HSBC_s15
                                 + 0.00491567534133726 w_k10_HSBC_s16
                                 + 0.00977326275374039 w_k10_HSBC_s17
                                 + 0.00719002301104456 w_k10_HSBC_s18
                                 + 0.00518204521968827 w_k10_HSBC_s19
                                 + 0.0188141637033941 w_k10_HSBC_s20
                                 + 0.0132401198586754 w_k10_HSBC_s21
                                 + 0.00914510907952506 w_k10_HSBC_s22
                                 + 0.0333401364585869 w_k10_HSBC_s23
                                 + 0.00861182106105592 w_k10_HSBC_s24
                                 + 0.0170597440736945 w_k10_HSBC_s25
                                 + 0.00931901566361393 w_k10_HSBC_s26
                                 + 0.0168505237064809 w_k10_HSBC_s27
                                 + 0.00967941580552924 w_k10_HSBC_s28
                                 + 0.0129088757926824 w_k10_HSBC_s29
                                 + 0.00816068493625416 w_k10_HSBC_s30
                                 + 0.00590403835876513 w_k10_HSBC_s31
                                 + 0.0102811639025851 w_k10_HSBC_s32
                                 + 0.0236794760580905 w_k10_HSBC_s33
                                 + 0.00467869071429587 w_k10_HSBC_s34
                                 + 0.00574704798370124 w_k10_HSBC_s35
                                 + 0.00503386722811171 w_k10_HSBC_s36
                                 + 0.00853138798625733 w_k10_HSBC_s37
                                 + 0.00637469686209976 w_k10_HSBC_s38
                                 + 0.00833358637470027 w_k10_HSBC_s39
                                 + 0.0101251682092274 w_k10_HSBC_s40
                                 + 0.00596697013473783 w_k10_HSBC_s41
                                 + 0.0042382843710921 w_k10_HSBC_s42
                                 + 0.00953147150066937 w_k10_HSBC_s43
                                 + 0.00562106407706648 w_k10_HSBC_s44
                                 + 0.0107066388647499 w_k10_HSBC_s45
                                 + 0.0113823699149865 w_k10_HSBC_s46
                                 + 0.00500295605227276 w_k10_HSBC_s47
                                 + 0.0129247333742773 w_k10_HSBC_s48
                                 + 0.00450583855108644 w_k10_HSBC_s49
                                 + 0.00703922585475469 w_k10_HSBC_s50
                                 + 0.00282220004672804 w_k10_HSBC_s51
                                 + 0.00244798220066938 w_k10_HSBC_s52
                                 + 0.00373254455807339 w_k10_HSBC_s53
                                 + 0.00290383076119797 w_k10_HSBC_s54
                                 + 0.00794320087373023 w_k10_HSBC_s55
                                 + 0.00324714294481631 w_k10_HSBC_s56
                                 + 0.00456834010946926 w_k10_HSBC_s57
                                 + 0.00542022806743839 w_k10_HSBC_s58
                                 + 0.00428671726598451 w_k10_HSBC_s59
                                 + 0.00422117765987372 w_k10_HSBC_s60
                                 + 0.00693233448516891 w_k10_HSBC_s61
                                 + 0.00480804600330785 w_k10_HSBC_s62
                                 + 0.0130347913261896 w_k10_HSBC_s63
                                 + 0.00387419864286091 w_k10_HSBC_s64
                                 + 0.0126389187849508 w_k10_HSBC_s65
                                 + 0.0064280415772944 w_k10_HSBC_s66
                                 + 0.00504048186361989 w_k10_HSBC_s67
                                 + 0.0111716419650492 w_k10_HSBC_s68
                                 + 0.00496100377884876 w_k10_HSBC_s69
                                 + 0.00860470518564553 w_k10_HSBC_s70
                                 + 0.00713065994331491 w_k10_HSBC_s71
                                 + 0.0105019647518113 w_k10_HSBC_s72
                                 + 0.00383733331637386 w_k10_HSBC_s73
                                 + 0.00762590816949307 w_k10_HSBC_s74
                                 + 0.0051616192501198 w_k10_HSBC_s75
                                 + 0.0123956121063885 w_k10_HSBC_s76
                                 + 0.00461310070532224 w_k10_HSBC_s77
                                 + 0.0142013841367196 w_k10_HSBC_s78
                                 + 0.00756798683319867 w_k10_HSBC_s79
                                 + 0.0231299819043574 w_k10_HSBC_s80
                                 + 0.042530233791355 w_k10_HSBC_s81
                                 + 0.0116614629302624 w_k10_HSBC_s82
                                 + 0.0141384853899811 w_k10_HSBC_s83
                                 + 0.0111262728075386 w_k10_HSBC_s84
                                 + 0.0186802814234175 w_k10_HSBC_s85
                                 + 0.0120354262241474 w_k10_HSBC_s86
                                 + 0.010536680862617 w_k10_HSBC_s87
                                 + 0.0219625063262825 w_k10_HSBC_s88
                                 + 0.0113622640398066 w_k10_HSBC_s89
                                 + 0.011391009115468 w_k10_HSBC_s90
                                 + 0.0111947985201637 w_k10_HSBC_s91
                                 + 0.00509187504189509 w_k10_HSBC_s92
                                 + 0.00340824247264298 w_k10_HSBC_s93
                                 + 0.00641953211380249 w_k10_HSBC_s94
                                 + 0.0150919610662767 w_k10_HSBC_s95
                                 + 0.00963659162423624 w_k10_HSBC_s96
                                 + 0.0154353849248381 w_k10_HSBC_s97
                                 + 0.0133242208684257 w_k10_HSBC_s98
                                 + 0.00556848867247983 w_k10_HSBC_s99
                                 + 0.0356540124548449 w_k10_VOD_s00
                                 + 0.0111993615642829 w_k10_VOD_s01
                                 + 0.0111205390081002 w_k10_VOD_s02
                                 + 0.0110322956826667 w_k10_VOD_s03
                                 + 0.00747445805342767 w_k10_VOD_s04
                                 + 0.013344849426456 w_k10_VOD_s05
                                 + 0.0150095269571256 w_k10_VOD_s06
                                 + 0.00904438273711905 w_k10_VOD_s07
                                 + 0.00735898322670826 w_k10_VOD_s08
                                 + 0.013787518957921 w_k10_VOD_s09
                                 + 0.0124811610211624 w_k10_VOD_s10
                                 + 0.00740732252452861 w_k10_VOD_s11
                                 + 0.0169421660941583 w_k10_VOD_s12
                                 + 0.00575656053265018 w_k10_VOD_s13
                                 + 0.019723934052411 w_k10_VOD_s14
                                 + 0.00734821658160408 w_k10_VOD_s15
                                 + 0.00505511735146282 w_k10_VOD_s16
                                 + 0.00836800819895285 w_k10_VOD_s17
                                 + 0.0079439093502779 w_k10_VOD_s18
                                 + 0.00452766171476083 w_k10_VOD_s19
                                 + 0.0184105749656928 w_k10_VOD_s20
                                 + 0.0139701477017553 w_k10_VOD_s21
                                 + 0.00991716496725182 w_k10_VOD_s22
                                 + 0.0330042455712911 w_k10_VOD_s23
                                 + 0.00865501153317412 w_k10_VOD_s24
                                 + 0.0145204816611835 w_k10_VOD_s25
                                 + 0.0086020945427362 w_k10_VOD_s26
                                 + 0.0168018259096255 w_k10_VOD_s27
                                 + 0.00880180656070869 w_k10_VOD_s28
                                 + 0.011186604069058 w_k10_VOD_s29
                                 + 0.00841625512694267 w_k10_VOD_s30
                                 + 0.00566934529426384 w_k10_VOD_s31
                                 + 0.00978513337673611 w_k10_VOD_s32
                                 + 0.0231083077087615 w_k10_VOD_s33
                                 + 0.00489890140968045 w_k10_VOD_s34
                                 + 0.00572522588417919 w_k10_VOD_s35
                                 + 0.00486890259194196 w_k10_VOD_s36
                                 + 0.007585277275129 w_k10_VOD_s37
                                 + 0.00604790528973263 w_k10_VOD_s38
                                 + 0.00856841124789183 w_k10_VOD_s39
                                 + 0.0101922249494989 w_k10_VOD_s40
                                 + 0.00645036545904685 w_k10_VOD_s41
                                 + 0.00463595106086277 w_k10_VOD_s42
                                 + 0.0097873437275958 w_k10_VOD_s43
                                 + 0.0055428774880385 w_k10_VOD_s44
                                 + 0.00959985710251729 w_k10_VOD_s45
                                 + 0.0114645566692868 w_k10_VOD_s46
                                 + 0.00519840719718477 w_k10_VOD_s47
                                 + 0.0127244680412629 w_k10_VOD_s48
                                 + 0.00411604662317137 w_k10_VOD_s49
                                 + 0.00720506635784301 w_k10_VOD_s50
                                 + 0.00298770143578576 w_k10_VOD_s51
                                 + 0.00239759973790767 w_k10_VOD_s52
                                 + 0.00371920369876869 w_k10_VOD_s53
                                 + 0.00274685700669962 w_k10_VOD_s54
                                 + 0.00833426755978704 w_k10_VOD_s55
                                 + 0.00334909782075685 w_k10_VOD_s56
                                 + 0.00470943505138322 w_k10_VOD_s57
                                 + 0.00550031914376832 w_k10_VOD_s58
                                 + 0.00424124098872216 w_k10_VOD_s59
                                 + 0.00460694010321662 w_k10_VOD_s60
                                 + 0.00692157365712821 w_k10_VOD_s61
                                 + 0.00509851393263499 w_k10_VOD_s62
                                 + 0.0129692531245863 w_k10_VOD_s63
                                 + 0.00394273598964857 w_k10_VOD_s64
                                 + 0.0142160568408705 w_k10_VOD_s65
                                 + 0.006200776917387 w_k10_VOD_s66
                                 + 0.00421612669715903 w_k10_VOD_s67
                                 + 0.00996547933581728 w_k10_VOD_s68
                                 + 0.00514712161624732 w_k10_VOD_s69
                                 + 0.0070677247348696 w_k10_VOD_s70
                                 + 0.00701087306635934 w_k10_VOD_s71
                                 + 0.00957396335180998 w_k10_VOD_s72
                                 + 0.00374288457917943 w_k10_VOD_s73
                                 + 0.00705828253177962 w_k10_VOD_s74
                                 + 0.00496732358399581 w_k10_VOD_s75
                                 + 0.0112461423005866 w_k10_VOD_s76
                                 + 0.00446105056726979 w_k10_VOD_s77
                                 + 0.014529613039686 w_k10_VOD_s78
                                 + 0.00795288957996926 w_k10_VOD_s79
                                 + 0.0219577223382005 w_k10_VOD_s80
                                 + 0.0446852484224719 w_k10_VOD_s81
                                 + 0.0102405239537341 w_k10_VOD_s82
                                 + 0.0158161359199424 w_k10_VOD_s83
                                 + 0.0108174637997489 w_k10_VOD_s84
                                 + 0.0194182241176361 w_k10_VOD_s85
                                 + 0.0116140338533213 w_k10_VOD_s86
                                 + 0.0109790282685492 w_k10_VOD_s87
                                 + 0.0217166147598571 w_k10_VOD_s88
                                 + 0.00950175681390665 w_k10_VOD_s89
                                 + 0.010473375251397 w_k10_VOD_s90
                                 + 0.00968762982533747 w_k10_VOD_s91
                                 + 0.00522158212575066 w_k10_VOD_s92
                                 + 0.00312032614729248 w_k10_VOD_s93
                                 + 0.00677345445596303 w_k10_VOD_s94
                                 + 0.0125648854679869 w_k10_VOD_s95
                                 + 0.00943832179048101 w_k10_VOD_s96
                                 + 0.0150563608778132 w_k10_VOD_s97
                                 + 0.0129125246687747 w_k10_VOD_s98
                                 + 0.00553315922080681 w_k10_VOD_s99
                                 + 0.0352609505346778 w_k10_BP_s00
                                 + 0.0106388968962883 w_k10_BP_s01
                                 + 0.0105971999231711 w_k10_BP_s02
                                 + 0.0112605200340466 w_k10_BP_s03
                                 + 0.00846962181294659 w_k10_BP_s04
                                 + 0.0140092440411108 w_k10_BP_s05
                                 + 0.0157446865670542 w_k10_BP_s06
                                 + 0.0086773684032607 w_k10_BP_s07
                                 + 0.00750635941708016 w_k10_BP_s08
                                 + 0.0141745058880039 w_k10_BP_s09
                                 + 0.0128778930207791 w_k10_BP_s10
                                 + 0.00729324411651493 w_k10_BP_s11
                                 + 0.0151445873411759 w_k10_BP_s12
                                 + 0.00624988673416584 w_k10_BP_s13
                                 + 0.0183425605182876 w_k10_BP_s14
                                 + 0.00712363806369832 w_k10_BP_s15
                                 + 0.00527966096969887 w_k10_BP_s16
                                 + 0.00965823088270341 w_k10_BP_s17
                                 + 0.00716516750902919 w_k10_BP_s18
                                 + 0.00482204959325715 w_k10_BP_s19
                                 + 0.0191517605243644 w_k10_BP_s20
                                 + 0.012860923190416 w_k10_BP_s21
                                 + 0.0105097332512335 w_k10_BP_s22
                                 + 0.0309640171234412 w_k10_BP_s23
                                 + 0.00880276020221521 w_k10_BP_s24
                                 + 0.0163100528008043 w_k10_BP_s25
                                 + 0.010159973443156 w_k10_BP_s26
                                 + 0.0179182310250224 w_k10_BP_s27
                                 + 0.00919477052451786 w_k10_BP_s28
                                 + 0.0119758392255694 w_k10_BP_s29
                                 + 0.00855365235998426 w_k10_BP_s30
                                 + 0.00542285846147271 w_k10_BP_s31
                                 + 0.0111314756500276 w_k10_BP_s32
                                 + 0.0228167101866741 w_k10_BP_s33
                                 + 0.00456439597385933 w_k10_BP_s34
                                 + 0.00552252525172422 w_k10_BP_s35
                                 + 0.00515655507670081 w_k10_BP_s36
                                 + 0.00796303124130363 w_k10_BP_s37
                                 + 0.00571489456613397 w_k10_BP_s38
                                 + 0.007724454107226 w_k10_BP_s39
                                 + 0.0109083099614793 w_k10_BP_s40
                                 + 0.0062649267547916 w_k10_BP_s41
                                 + 0.00427950305365859 w_k10_BP_s42
                                 + 0.00858929555126584 w_k10_BP_s43
                                 + 0.00594798492122104 w_k10_BP_s44
                                 + 0.00984462592496352 w_k10_BP_s45
                                 + 0.0113597958841659 w_k10_BP_s46
                                 + 0.00528172154090924 w_k10_BP_s47
                                 + 0.0132378447240082 w_k10_BP_s48
                                 + 0.00425225902747189 w_k10_BP_s49
                                 + 0.00772931746248269 w_k10_BP_s50
                                 + 0.00293475333691077 w_k10_BP_s51
                                 + 0.00245362391180542 w_k10_BP_s52
                                 + 0.00385502919129201 w_k10_BP_s53
                                 + 0.00309479441387741 w_k10_BP_s54
                                 + 0.00748052944753987 w_k10_BP_s55
                                 + 0.00323993992514702 w_k10_BP_s56
                                 + 0.00474291617187053 w_k10_BP_s57
                                 + 0.00513328915432966 w_k10_BP_s58
                                 + 0.00431269738521905 w_k10_BP_s59
                                 + 0.00407588585715864 w_k10_BP_s60
                                 + 0.0067508463749138 w_k10_BP_s61
                                 + 0.00518112879498094 w_k10_BP_s62
                                 + 0.0131338917817314 w_k10_BP_s63
                                 + 0.00360834114456687 w_k10_BP_s64
                                 + 0.0149403258149428 w_k10_BP_s65
                                 + 0.006305998208175 w_k10_BP_s66
                                 + 0.00459091431761863 w_k10_BP_s67
                                 + 0.0109732623900683 w_k10_BP_s68
                                 + 0.0046640259944951 w_k10_BP_s69
                                 + 0.00762797343590722 w_k10_BP_s70
                                 + 0.00716415151394035 w_k10_BP_s71
                                 + 0.0105459171551544 w_k10_BP_s72
                                 + 0.0036543380412301 w_k10_BP_s73
                                 + 0.00675840847933012 w_k10_BP_s74
                                 + 0.00527742986760528 w_k10_BP_s75
                                 + 0.0113957270074553 w_k10_BP_s76
                                 + 0.00492908170524641 w_k10_BP_s77
                                 + 0.0149255565728773 w_k10_BP_s78
                                 + 0.00801799708698615 w_k10_BP_s79
                                 + 0.0223026198402472 w_k10_BP_s80
                                 + 0.0431729852225587 w_k10_BP_s81
                                 + 0.0118824500631625 w_k10_BP_s82
                                 + 0.0137064069413554 w_k10_BP_s83
                                 + 0.0116159920516694 w_k10_BP_s84
                                 + 0.018410451839624 w_k10_BP_s85
                                 + 0.0119719293864267 w_k10_BP_s86
                                 + 0.0106301323116855 w_k10_BP_s87
                                 + 0.0211195066009201 w_k10_BP_s88
                                 + 0.0114910245312201 w_k10_BP_s89
                                 + 0.0108390047888051 w_k10_BP_s90
                                 + 0.00986114340233818 w_k10_BP_s91
                                 + 0.00503925738287779 w_k10_BP_s92
                                 + 0.00339695403195295 w_k10_BP_s93
                                 + 0.00707164622569942 w_k10_BP_s94
                                 + 0.0152244352358164 w_k10_BP_s95
                                 + 0.00977974189280091 w_k10_BP_s96
                                 + 0.0153610623991217 w_k10_BP_s97
                                 + 0.0114632199840774 w_k10_BP_s98
                                 + 0.00558176789947975 w_k10_BP_s99
                                 + 0.0339275589424373 w_k10_GSK_s00
                                 + 0.0105722763848836 w_k10_GSK_s01
                                 + 0.0111053022075833 w_k10_GSK_s02
                                 + 0.0108635818558577 w_k10_GSK_s03
                                 + 0.00806461869037389 w_k10_GSK_s04
                                 + 0.014643513391379 w_k10_GSK_s05
                                 + 0.0158725642006812 w_k10_GSK_s06
                                 + 0.00874724726943051 w_k10_GSK_s07
                                 + 0.00840094790034336 w_k10_GSK_s08
                                 + 0.0138349701534989 w_k10_GSK_s09
                                 + 0.01204251162391 w_k10_GSK_s10
                                 + 0.00826386987401936 w_k10_GSK_s11
                                 + 0.016014367667226 w_k10_GSK_s12
                                 + 0.00583366104225507 w_k10_GSK_s13
                                 + 0.019677857166796 w_k10_GSK_s14
                                 + 0.00770708771997828 w_k10_GSK_s15
                                 + 0.00501238019432599 w_k10_GSK_s16
                                 + 0.0104534467769661 w_k10_GSK_s17
                                 + 0.0075602191326768 w_k10_GSK_s18
                                 + 0.00449178990087977 w_k10_GSK_s19
                                 + 0.0184372106690847 w_k10_GSK_s20
                                 + 0.013026127263654 w_k10_GSK_s21
                                 + 0.0102843175583077 w_k10_GSK_s22
                                 + 0.0336422269983991 w_k10_GSK_s23
                                 + 0.00808876667628019 w_k10_GSK_s24
                                 + 0.0161667722839131 w_k10_GSK_s25
                                 + 0.00919517726848107 w_k10_GSK_s26
                                 + 0.018881592206431 w_k10_GSK_s27
                                 + 0.00906479830769986 w_k10_GSK_s28
                                 + 0.0121156893170628 w_k10_GSK_s29
                                 + 0.00905200848226371 w_k10_GSK_s30
                                 + 0.00547596826360069 w_k10_GSK_s31
                                 + 0.0104665833656904 w_k10_GSK_s32
                                 + 0.0221588716898779 w_k10_GSK_s33
                                 + 0.00488791139971032 w_k10_GSK_s34
                                 + 0.00492380082313013 w_k10_GSK_s35
                                 + 0.00507026929279433 w_k10_GSK_s36
                                 + 0.00854279638634746 w_k10_GSK_s37
                                 + 0.00628012556810651 w_k10_GSK_s38
                                 + 0.00767778565422756 w_k10_GSK_s39
                                 + 0.0106333239025132 w_k10_GSK_s40
                                 + 0.00688655208601386 w_k10_GSK_s41
                                 + 0.0041996632906102 w_k10_GSK_s42
                                 + 0.00895755420412369 w_k10_GSK_s43
                                 + 0.0053394034299756 w_k10_GSK_s44
                                 + 0.0103247307738899 w_k10_GSK_s45
                                 + 0.0113073667343464 w_k10_GSK_s46
                                 + 0.00494507608058744 w_k10_GSK_s47
                                 + 0.013240250630021 w_k10_GSK_s48
                                 + 0.00404644526494206 w_k10_GSK_s49
                                 + 0.00653680699542878 w_k10_GSK_s50
                                 + 0.00270303407682112 w_k10_GSK_s51
                                 + 0.0024024211148927 w_k10_GSK_s52
                                 + 0.00404759577401459 w_k10_GSK_s53
                                 + 0.00320482720846387 w_k10_GSK_s54
                                 + 0.00785086134483924 w_k10_GSK_s55
                                 + 0.0032395496959414 w_k10_GSK_s56
                                 + 0.00489314718965141 w_k10_GSK_s57
                                 + 0.0050682491015154 w_k10_GSK_s58
                                 + 0.00444866571016843 w_k10_GSK_s59
                                 + 0.00415915945197681 w_k10_GSK_s60
                                 + 0.0062712858291184 w_k10_GSK_s61
                                 + 0.00499943567876992 w_k10_GSK_s62
                                 + 0.0128543346468633 w_k10_GSK_s63
                                 + 0.00340585929922989 w_k10_GSK_s64
                                 + 0.0146049545354355 w_k10_GSK_s65
                                 + 0.006835633146432 w_k10_GSK_s66
                                 + 0.00466391140079405 w_k10_GSK_s67
                                 + 0.0101495407626601 w_k10_GSK_s68
                                 + 0.00486731177722947 w_k10_GSK_s69
                                 + 0.00772418606681607 w_k10_GSK_s70
                                 + 0.00775087749091732 w_k10_GSK_s71
                                 + 0.0109972079165997 w_k10_GSK_s72
                                 + 0.0034607396703241 w_k10_GSK_s73
                                 + 0.00675441232500287 w_k10_GSK_s74
                                 + 0.00498232250737405 w_k10_GSK_s75
                                 + 0.0120316324735864 w_k10_GSK_s76
                                 + 0.00476907676848062 w_k10_GSK_s77
                                 + 0.0144102404616836 w_k10_GSK_s78
                                 + 0.00743299803233126 w_k10_GSK_s79
                                 + 0.0226818683635172 w_k10_GSK_s80
                                 + 0.047148436432568 w_k10_GSK_s81
                                 + 0.0120935991267638 w_k10_GSK_s82
                                 + 0.0123684047076365 w_k10_GSK_s83
                                 + 0.011260079335927 w_k10_GSK_s84
                                 + 0.018301418581775 w_k10_GSK_s85
                                 + 0.0118225639225665 w_k10_GSK_s86
                                 + 0.0104139554194894 w_k10_GSK_s87
                                 + 0.0216625218371915 w_k10_GSK_s88
                                 + 0.0110059586025665 w_k10_GSK_s89
                                 + 0.0115199428264649 w_k10_GSK_s90
                                 + 0.0109666956460071 w_k10_GSK_s91
                                 + 0.00449174152890902 w_k10_GSK_s92
                                 + 0.00325275880453614 w_k10_GSK_s93
                                 + 0.00745640845715975 w_k10_GSK_s94
                                 + 0.0148259386848762 w_k10_GSK_s95
                                 + 0.010453021846322 w_k10_GSK_s96
                                 + 0.0154497990267474 w_k10_GSK_s97
                                 + 0.0128952135529117 w_k10_GSK_s98
                                 + 0.00569742754140604 w_k10_GSK_s99
                                 + 0.0323295210721004 w_k10_AZN_s00
                                 + 0.0110520404821024 w_k10_AZN_s01
                                 + 0.0121734095012671 w_k10_AZN_s02
                                 + 0.0115849151167912 w_k10_AZN_s03
                                 + 0.00798552567803254 w_k10_AZN_s04
                                 + 0.0138299618806154 w_k10_AZN_s05
                                 + 0.0156357583652013 w_k10_AZN_s06
                                 + 0.00960484896255439 w_k10_AZN_s07
                                 + 0.00783593410144361 w_k10_AZN_s08
                                 + 0.0135027466434821 w_k10_AZN_s09
                                 + 0.0116704717574736 w_k10_AZN_s10
                                 + 0.00813952045567429 w_k10_AZN_s11
                                 + 0.0161982293954359 w_k10_AZN_s12
                                 + 0.00625272270651619 w_k10_AZN_s13
                                 + 0.0195755979446462 w_k10_AZN_s14
                                 + 0.00802266710894968 w_k10_AZN_s15
                                 + 0.00487272756171294 w_k10_AZN_s16
                                 + 0.00863824634654946 w_k10_AZN_s17
                                 + 0.00790030347370948 w_k10_AZN_s18
                                 + 0.00447725148961341 w_k10_AZN_s19
                                 + 0.0170875519958468 w_k10_AZN_s20
                                 + 0.0134853946287079 w_k10_AZN_s21
                                 + 0.00964215327762222 w_k10_AZN_s22
                                 + 0.0322909226072693 w_k10_AZN_s23
                                 + 0.00849616635023339 w_k10_AZN_s24
                                 + 0.0153532760455001 w_k10_AZN_s25
                                 + 0.00905742929779532 w_k10_AZN_s26
                                 + 0.0178779359320995 w_k10_AZN_s27
                                 + 0.00880573590911563 w_k10_AZN_s28
                                 + 0.0125081107050701 w_k10_AZN_s29
                                 + 0.00851763674611822 w_k10_AZN_s30
                                 + 0.00547248470140483 w_k10_AZN_s31
                                 + 0.00981275313389838 w_k10_AZN_s32
                                 + 0.0215563326176342 w_k10_AZN_s33
                                 + 0.00535888063232873 w_k10_AZN_s34
                                 + 0.00529767687540683 w_k10_AZN_s35
                                 + 0.00551092096418552 w_k10_AZN_s36
                                 + 0.00791263167382736 w_k10_AZN_s37
                                 + 0.00555901745772801 w_k10_AZN_s38
                                 + 0.00781448258406447 w_k10_AZN_s39
                                 + 0.0104301473047143 w_k10_AZN_s40
                                 + 0.00675046219528986 w_k10_AZN_s41
                                 + 0.00401598750881561 w_k10_AZN_s42
                                 + 0.00946419776678254 w_k10_AZN_s43
                                 + 0.00528350935102861 w_k10_AZN_s44
                                 + 0.0102685695578746 w_k10_AZN_s45
                                 + 0.0103967142810913 w_k10_AZN_s46
                                 + 0.0051838060932615 w_k10_AZN_s47
                                 + 0.0131974380170887 w_k10_AZN_s48
                                 + 0.00388683125432745 w_k10_AZN_s49
                                 + 0.00675306077850009 w_k10_AZN_s50
                                 + 0.00282572892151413 w_k10_AZN_s51
                                 + 0.00240021759305446 w_k10_AZN_s52
                                 + 0.00379989824555367 w_k10_AZN_s53
                                 + 0.00296885193164879 w_k10_AZN_s54
                                 + 0.00811322298326712 w_k10_AZN_s55
                                 + 0.00312091940875327 w_k10_AZN_s56
                                 + 0.00491236110573685 w_k10_AZN_s57
                                 + 0.0050814445462409 w_k10_AZN_s58
                                 + 0.00425869563183988 w_k10_AZN_s59
                                 + 0.00451766506034658 w_k10_AZN_s60
                                 + 0.00646113008624565 w_k10_AZN_s61
                                 + 0.00497140762487949 w_k10_AZN_s62
                                 + 0.0121353219520541 w_k10_AZN_s63
                                 + 0.00352566042211248 w_k10_AZN_s64
                                 + 0.0134833545683662 w_k10_AZN_s65
                                 + 0.0064498455692856 w_k10_AZN_s66
                                 + 0.00446028238849492 w_k10_AZN_s67
                                 + 0.0106389426056865 w_k10_AZN_s68
                                 + 0.00499039764523616 w_k10_AZN_s69
                                 + 0.00750046751352459 w_k10_AZN_s70
                                 + 0.00795763326824851 w_k10_AZN_s71
                                 + 0.00981783251981894 w_k10_AZN_s72
                                 + 0.00365462044285137 w_k10_AZN_s73
                                 + 0.0069862240374141 w_k10_AZN_s74
                                 + 0.00510505509573922 w_k10_AZN_s75
                                 + 0.0123091823297685 w_k10_AZN_s76
                                 + 0.00443072917481152 w_k10_AZN_s77
                                 + 0.0145982270536754 w_k10_AZN_s78
                                 + 0.00808060051881388 w_k10_AZN_s79
                                 + 0.0222363908167041 w_k10_AZN_s80
                                 + 0.0436139830208015 w_k10_AZN_s81
                                 + 0.0108448763455132 w_k10_AZN_s82
                                 + 0.0136525066715598 w_k10_AZN_s83
                                 + 0.0106725429897541 w_k10_AZN_s84
                                 + 0.0199123037326501 w_k10_AZN_s85
                                 + 0.0116054027373715 w_k10_AZN_s86
                                 + 0.0113982983328791 w_k10_AZN_s87
                                 + 0.0212097577951874 w_k10_AZN_s88
                                 + 0.0106093510205586 w_k10_AZN_s89
                                 + 0.0119965272733495 w_k10_AZN_s90
                                 + 0.0101467900719183 w_k10_AZN_s91
                                 + 0.00494992220012263 w_k10_AZN_s92
                                 + 0.00331556186153863 w_k10_AZN_s93
                                 + 0.00735273941974883 w_k10_AZN_s94
                                 + 0.0142471446280304 w_k10_AZN_s95
                                 + 0.0102601947858817 w_k10_AZN_s96
                                 + 0.0162030289004464 w_k10_AZN_s97
                                 + 0.0124055848454616 w_k10_AZN_s98
                                 + 0.00549153986585845 w_k10_AZN_s99
                                 + 0.0347630033402724 w_k10_RIO_s00
                                 + 0.0111144694674442 w_k10_RIO_s01
                                 + 0.0105291131238487 w_k10_RIO_s02
                                 + 0.0106755588312744 w_k10_RIO_s03
                                 + 0.00795910665677574 w_k10_RIO_s04
                                 + 0.0136746551579716 w_k10_RIO_s05
                                 + 0.0152201243840502 w_k10_RIO_s06
                                 + 0.00868860490394618 w_k10_RIO_s07
                                 + 0.00742052748307715 w_k10_RIO_s08
                                 + 0.0137124950939014 w_k10_RIO_s09
                                 + 0.0123339997669555 w_k10_RIO_s10
                                 + 0.00751689533666656 w_k10_RIO_s11
                                 + 0.0171512775035007 w_k10_RIO_s12
                                 + 0.00559079520564668 w_k10_RIO_s13
                                 + 0.0183885023527891 w_k10_RIO_s14
                                 + 0.00728281234263029 w_k10_RIO_s15
                                 + 0.00517451817662879 w_k10_RIO_s16
                                 + 0.0100590020826644 w_k10_RIO_s17
                                 + 0.00805430742564437 w_k10_RIO_s18
                                 + 0.00497519569170041 w_k10_RIO_s19
                                 + 0.0191932116878698 w_k10_RIO_s20
                                 + 0.0136314854931315 w_k10_RIO_s21
                                 + 0.0105556383055453 w_k10_RIO_s22
                                 + 0.0336507515000998 w_k10_RIO_s23
                                 + 0.00841934496568446 w_k10_RIO_s24
                                 + 0.0158752217995131 w_k10_RIO_s25
                                 + 0.0105133712354792 w_k10_RIO_s26
                                 + 0.0187426751771888 w_k10_RIO_s27
                                 + 0.00879777525143778 w_k10_RIO_s28
                                 + 0.0119755928553247 w_k10_RIO_s29
                                 + 0.00828617859525226 w_k10_RIO_s30
                                 + 0.00599016632790966 w_k10_RIO_s31
                                 + 0.0112922364145771 w_k10_RIO_s32
                                 + 0.0234649444062351 w_k10_RIO_s33
                                 + 0.00454673110427173 w_k10_RIO_s34
                                 + 0.00550338766765293 w_k10_RIO_s35
                                 + 0.0051300074611839 w_k10_RIO_s36
                                 + 0.00796769017373127 w_k10_RIO_s37
                                 + 0.00588860305979461 w_k10_RIO_s38
                                 + 0.00831396511377335 w_k10_RIO_s39
                                 + 0.0113277802322931 w_k10_RIO_s40
                                 + 0.00608520712535665 w_k10_RIO_s41
                                 + 0.00468406428822123 w_k10_RIO_s42
                                 + 0.00863883784951141 w_k10_RIO_s43
                                 + 0.00553492424380546 w_k10_RIO_s44
                                 + 0.0101378643429264 w_k10_RIO_s45
                                 + 0.0111645301608933 w_k10_RIO_s46
                                 + 0.00519004913121773 w_k10_RIO_s47
                                 + 0.0116027343218723 w_k10_RIO_s48
                                 + 0.00408150466187818 w_k10_RIO_s49
                                 + 0.00715182544241621 w_k10_RIO_s50
                                 + 0.00283087750707453 w_k10_RIO_s51
                                 + 0.00247927150624675 w_k10_RIO_s52
                                 + 0.00401042868993697 w_k10_RIO_s53
                                 + 0.00305631432030677 w_k10_RIO_s54
                                 + 0.0078417707221013 w_k10_RIO_s55
                                 + 0.00327510616544423 w_k10_RIO_s56
                                 + 0.0044252335227071 w_k10_RIO_s57
                                 + 0.00538066488485237 w_k10_RIO_s58
                                 + 0.00451615051011971 w_k10_RIO_s59
                                 + 0.00444467647522327 w_k10_RIO_s60
                                 + 0.00661421275329019 w_k10_RIO_s61
                                 + 0.0051761807604969 w_k10_RIO_s62
                                 + 0.0132737197845068 w_k10_RIO_s63
                                 + 0.0038768212042209 w_k10_RIO_s64
                                 + 0.0148490836640514 w_k10_RIO_s65
                                 + 0.0062443248606684 w_k10_RIO_s66
                                 + 0.00471047791639856 w_k10_RIO_s67
                                 + 0.0104190944789688 w_k10_RIO_s68
                                 + 0.00471055070467048 w_k10_RIO_s69
                                 + 0.00766752126203463 w_k10_RIO_s70
                                 + 0.00682678911113371 w_k10_RIO_s71
                                 + 0.00965190506310994 w_k10_RIO_s72
                                 + 0.00402564632222681 w_k10_RIO_s73
                                 + 0.00697971094474335 w_k10_RIO_s74
                                 + 0.00550839049568745 w_k10_RIO_s75
                                 + 0.0107015399003797 w_k10_RIO_s76
                                 + 0.00473900783069958 w_k10_RIO_s77
                                 + 0.016181979953184 w_k10_RIO_s78
                                 + 0.00741692223908318 w_k10_RIO_s79
                                 + 0.0224191008726914 w_k10_RIO_s80
                                 + 0.0428842414795282 w_k10_RIO_s81
                                 + 0.0118424109576097 w_k10_RIO_s82
                                 + 0.0137494690712534 w_k10_RIO_s83
                                 + 0.0108553854421724 w_k10_RIO_s84
                                 + 0.0181446251035047 w_k10_RIO_s85
                                 + 0.0120409397205955 w_k10_RIO_s86
                                 + 0.00986367318997909 w_k10_RIO_s87
                                 + 0.0208942577672482 w_k10_RIO_s88
                                 + 0.0110651581018412 w_k10_RIO_s89
                                 + 0.0103508763795634 w_k10_RIO_s90
                                 + 0.010559445363085 w_k10_RIO_s91
                                 + 0.0053455702059089 w_k10_RIO_s92
                                 + 0.00319030273730577 w_k10_RIO_s93
                                 + 0.00652166688642868 w_k10_RIO_s94
                                 + 0.0136302475932991 w_k10_RIO_s95
                                 + 0.0105821391398961 w_k10_RIO_s96
                                 + 0.0155901131633243 w_k10_RIO_s97
                                 + 0.0115811137283202 w_k10_RIO_s98
                                 + 0.00576927041096351 w_k10_RIO_s99
                                 + 0.0344680162606862 w_k10_BG_s00
                                 + 0.0107722620918572 w_k10_BG_s01
                                 + 0.0109989417760077 w_k10_BG_s02
                                 + 0.011435493872875 w_k10_BG_s03
                                 + 0.0076296983873869 w_k10_BG_s04
                                 + 0.0138279519936664 w_k10_BG_s05
                                 + 0.0153989302029712 w_k10_BG_s06
                                 + 0.00841674380157888 w_k10_BG_s07
                                 + 0.00734274307491632 w_k10_BG_s08
                                 + 0.0136478183937502 w_k10_BG_s09
                                 + 0.0124048752963356 w_k10_BG_s10
                                 + 0.00720248727795142 w_k10_BG_s11
                                 + 0.0159147292116256 w_k10_BG_s12
                                 + 0.00613694620033397 w_k10_BG_s13
                                 + 0.0177488842809024 w_k10_BG_s14
                                 + 0.00708878476950044 w_k10_BG_s15
                                 + 0.00500289635872173 w_k10_BG_s16
                                 + 0.00948740475191151 w_k10_BG_s17
                                 + 0.00770200121800422 w_k10_BG_s18
                                 + 0.0048181545552928 w_k10_BG_s19
                                 + 0.0190341787841732 w_k10_BG_s20
                                 + 0.0131501350625706 w_k10_BG_s21
                                 + 0.0105605221847039 w_k10_BG_s22
                                 + 0.0325450751952351 w_k10_BG_s23
                                 + 0.00847227088483744 w_k10_BG_s24
                                 + 0.0163286323710751 w_k10_BG_s25
                                 + 0.00955304221042875 w_k10_BG_s26
                                 + 0.0173444920123542 w_k10_BG_s27
                                 + 0.00934220861294848 w_k10_BG_s28
                                 + 0.012243815651555 w_k10_BG_s29
                                 + 0.00887166830665952 w_k10_BG_s30
                                 + 0.0055876836978251 w_k10_BG_s31
                                 + 0.0102363969341441 w_k10_BG_s32
                                 + 0.0223958758337283 w_k10_BG_s33
                                 + 0.00466087849645095 w_k10_BG_s34
                                 + 0.00535044284323434 w_k10_BG_s35
                                 + 0.00527805691004053 w_k10_BG_s36
                                 + 0.00842203660429874 w_k10_BG_s37
                                 + 0.00588032995413125 w_k10_BG_s38
                                 + 0.00836217437448342 w_k10_BG_s39
                                 + 0.0109263727246773 w_k10_BG_s40
                                 + 0.00658058365411636 w_k10_BG_s41
                                 + 0.00417185211790127 w_k10_BG_s42
                                 + 0.00900053274570789 w_k10_BG_s43
                                 + 0.00580359827264127 w_k10_BG_s44
                                 + 0.0101059233688187 w_k10_BG_s45
                                 + 0.0114989674618407 w_k10_BG_s46
                                 + 0.00508956991452814 w_k10_BG_s47
                                 + 0.0123052311154898 w_k10_BG_s48
                                 + 0.0043734343257189 w_k10_BG_s49
                                 + 0.00720678424324438 w_k10_BG_s50
                                 + 0.00278152622806914 w_k10_BG_s51
                                 + 0.00243308710767884 w_k10_BG_s52
                                 + 0.00376355287853228 w_k10_BG_s53
                                 + 0.00304881432728904 w_k10_BG_s54
                                 + 0.00760112518640674 w_k10_BG_s55
                                 + 0.00335983403409773 w_k10_BG_s56
                                 + 0.00473702224264699 w_k10_BG_s57
                                 + 0.00514506218161349 w_k10_BG_s58
                                 + 0.0043566994257176 w_k10_BG_s59
                                 + 0.00454344547987078 w_k10_BG_s60
                                 + 0.00608778768240069 w_k10_BG_s61
                                 + 0.00489642475224726 w_k10_BG_s62
                                 + 0.013122322221254 w_k10_BG_s63
                                 + 0.00341122969306697 w_k10_BG_s64
                                 + 0.0139303964314448 w_k10_BG_s65
                                 + 0.0063254992286646 w_k10_BG_s66
                                 + 0.00491057009061295 w_k10_BG_s67
                                 + 0.0106794004009171 w_k10_BG_s68
                                 + 0.00483512383742323 w_k10_BG_s69
                                 + 0.00780785380423612 w_k10_BG_s70
                                 + 0.00777688276083677 w_k10_BG_s71
                                 + 0.00962961173493369 w_k10_BG_s72
                                 + 0.00362953296063913 w_k10_BG_s73
                                 + 0.0068123066799375 w_k10_BG_s74
                                 + 0.00531146391508056 w_k10_BG_s75
                                 + 0.0113510776561071 w_k10_BG_s76
                                 + 0.0047962125711979 w_k10_BG_s77
                                 + 0.0143691979014724 w_k10_BG_s78
                                 + 0.00749935388519383 w_k10_BG_s79
                                 + 0.0219756599953495 w_k10_BG_s80
                                 + 0.0433563844421698 w_k10_BG_s81
                                 + 0.0109751430455301 w_k10_BG_s82
                                 + 0.0140257773845596 w_k10_BG_s83
                                 + 0.0110865896936253 w_k10_BG_s84
                                 + 0.018217540393243 w_k10_BG_s85
                                 + 0.0111943490954119 w_k10_BG_s86
                                 + 0.010609063971184 w_k10_BG_s87
                                 + 0.0203283759639934 w_k10_BG_s88
                                 + 0.0106445109407045 w_k10_BG_s89
                                 + 0.0110853028295745 w_k10_BG_s90
                                 + 0.0100810350322046 w_k10_BG_s91
                                 + 0.00512332693722872 w_k10_BG_s92
                                 + 0.00336740692027232 w_k10_BG_s93
                                 + 0.00718916975482933 w_k10_BG_s94
                                 + 0.0138838289194194 w_k10_BG_s95
                                 + 0.00996752769615676 w_k10_BG_s96
                                 + 0.0150622639661256 w_k10_BG_s97
                                 + 0.0124425351304202 w_k10_BG_s98
                                 + 0.00549830211945138 w_k10_BG_s99
                                 + 0.0350724952381554 w_k10_TSCO_s00
                                 + 0.0104864022410215 w_k10_TSCO_s01
                                 + 0.0104037811174228 w_k10_TSCO_s02
                                 + 0.0115560118658159 w_k10_TSCO_s03
                                 + 0.0082167998348663 w_k10_TSCO_s04
                                 + 0.0133409438350177 w_k10_TSCO_s05
                                 + 0.0153904366489872 w_k10_TSCO_s06
                                 + 0.007865084459006 w_k10_TSCO_s07
                                 + 0.00818355471977583 w_k10_TSCO_s08
                                 + 0.0138051578259487 w_k10_TSCO_s09
                                 + 0.0123140724543092 w_k10_TSCO_s10
                                 + 0.00715307236861824 w_k10_TSCO_s11
                                 + 0.0155848536259459 w_k10_TSCO_s12
                                 + 0.00615530209592957 w_k10_TSCO_s13
                                 + 0.0187211573925151 w_k10_TSCO_s14
                                 + 0.00795506378703066 w_k10_TSCO_s15
                                 + 0.00529110811758707 w_k10_TSCO_s16
                                 + 0.00937450233199317 w_k10_TSCO_s17
                                 + 0.00693373228808561 w_k10_TSCO_s18
                                 + 0.00454907432207331 w_k10_TSCO_s19
                                 + 0.0177756882813379 w_k10_TSCO_s20
                                 + 0.0125400203062106 w_k10_TSCO_s21
                                 + 0.0102596789922118 w_k10_TSCO_s22
                                 + 0.0302210845206757 w_k10_TSCO_s23
                                 + 0.00834572730507231 w_k10_TSCO_s24
                                 + 0.0165433858137922 w_k10_TSCO_s25
                                 + 0.00966310044240042 w_k10_TSCO_s26
                                 + 0.017487396184575 w_k10_TSCO_s27
                                 + 0.00966447752585725 w_k10_TSCO_s28
                                 + 0.0115777446713771 w_k10_TSCO_s29
                                 + 0.0087141457157078 w_k10_TSCO_s30
                                 + 0.00561507155380297 w_k10_TSCO_s31
                                 + 0.0106675415159305 w_k10_TSCO_s32
                                 + 0.0233642708389523 w_k10_TSCO_s33
                                 + 0.00455339737237744 w_k10_TSCO_s34
                                 + 0.0051360476886421 w_k10_TSCO_s35
                                 + 0.00537429340638934 w_k10_TSCO_s36
                                 + 0.00928016592800265 w_k10_TSCO_s37
                                 + 0.00576573670673693 w_k10_TSCO_s38
                                 + 0.00745636568229529 w_k10_TSCO_s39
                                 + 0.0117946868790111 w_k10_TSCO_s40
                                 + 0.0061888102207934 w_k10_TSCO_s41
                                 + 0.00420673417675536 w_k10_TSCO_s42
                                 + 0.00954250702044683 w_k10_TSCO_s43
                                 + 0.00544696216205109 w_k10_TSCO_s44
                                 + 0.0105998241932183 w_k10_TSCO_s45
                                 + 0.010721561466947 w_k10_TSCO_s46
                                 + 0.00506872378123136 w_k10_TSCO_s47
                                 + 0.0122768033260648 w_k10_TSCO_s48
                                 + 0.00389075510637717 w_k10_TSCO_s49
                                 + 0.00741280263227735 w_k10_TSCO_s50
                                 + 0.0027466810089979 w_k10_TSCO_s51
                                 + 0.00241556128371602 w_k10_TSCO_s52
                                 + 0.00401263450594701 w_k10_TSCO_s53
                                 + 0.00288478761774729 w_k10_TSCO_s54
                                 + 0.00867055499667466 w_k10_TSCO_s55
                                 + 0.00309000882248628 w_k10_TSCO_s56
                                 + 0.00457398171241986 w_k10_TSCO_s57
                                 + 0.00463677619404308 w_k10_TSCO_s58
                                 + 0.00459508263945163 w_k10_TSCO_s59
                                 + 0.00433017839542903 w_k10_TSCO_s60
                                 + 0.00620647731750909 w_k10_TSCO_s61
                                 + 0.00512161581324318 w_k10_TSCO_s62
                                 + 0.0123744322397242 w_k10_TSCO_s63
                                 + 0.00326432740076425 w_k10_TSCO_s64
                                 + 0.0153120833239972 w_k10_TSCO_s65
                                 + 0.007233271605432 w_k10_TSCO_s66
                                 + 0.00500836490734549 w_k10_TSCO_s67
                                 + 0.010865981827642 w_k10_TSCO_s68
                                 + 0.00463539821749298 w_k10_TSCO_s69
                                 + 0.00828615821353896 w_k10_TSCO_s70
                                 + 0.00772552701362977 w_k10_TSCO_s71
                                 + 0.0104490188299335 w_k10_TSCO_s72
                                 + 0.00340847575206193 w_k10_TSCO_s73
                                 + 0.00607154548618887 w_k10_TSCO_s74
                                 + 0.00522859023721032 w_k10_TSCO_s75
                                 + 0.0130081015210463 w_k10_TSCO_s76
                                 + 0.00485597608735885 w_k10_TSCO_s77
                                 + 0.0156015337682569 w_k10_TSCO_s78
                                 + 0.00801291872809519 w_k10_TSCO_s79
                                 + 0.0214527729391328 w_k10_TSCO_s80
                                 + 0.0454614997128267 w_k10_TSCO_s81
                                 + 0.0113432777338909 w_k10_TSCO_s82
                                 + 0.0150792133795761 w_k10_TSCO_s83
                                 + 0.0108000399619032 w_k10_TSCO_s84
                                 + 0.0173461169317621 w_k10_TSCO_s85
                                 + 0.0128351856477782 w_k10_TSCO_s86
                                 + 0.0101791410013854 w_k10_TSCO_s87
                                 + 0.0224359377944642 w_k10_TSCO_s88
                                 + 0.0115330907824928 w_k10_TSCO_s89
                                 + 0.0114832794544653 w_k10_TSCO_s90
                                 + 0.00982616253233971 w_k10_TSCO_s91
                                 + 0.00458355546326669 w_k10_TSCO_s92
                                 + 0.00327685788096824 w_k10_TSCO_s93
                                 + 0.00779373800441939 w_k10_TSCO_s94
                                 + 0.0133189567869873 w_k10_TSCO_s95
                                 + 0.0101918486693353 w_k10_TSCO_s96
                                 + 0.0148978812573878 w_k10_TSCO_s97
                                 + 0.0125872670821131 w_k10_TSCO_s98
                                 + 0.00518019033933373 w_k10_TSCO_s99
                                 + 0.0363590893343917 w_k10_AAL_s00
                                 + 0.0110465173410019 w_k10_AAL_s01
                                 + 0.0111839351412283 w_k10_AAL_s02
                                 + 0.0112215309009728 w_k10_AAL_s03
                                 + 0.00794078386755404 w_k10_AAL_s04
                                 + 0.0133853411940731 w_k10_AAL_s05
                                 + 0.0159416387542921 w_k10_AAL_s06
                                 + 0.00833633618141332 w_k10_AAL_s07
                                 + 0.00727063819126566 w_k10_AAL_s08
                                 + 0.0139468040051735 w_k10_AAL_s09
                                 + 0.0122561483893809 w_k10_AAL_s10
                                 + 0.00740670854624361 w_k10_AAL_s11
                                 + 0.0159403115117547 w_k10_AAL_s12
                                 + 0.00585979051698545 w_k10_AAL_s13
                                 + 0.0184629956018356 w_k10_AAL_s14
                                 + 0.00750693352566546 w_k10_AAL_s15
                                 + 0.00522079167505051 w_k10_AAL_s16
                                 + 0.0093922549482951 w_k10_AAL_s17
                                 + 0.00782974526557314 w_k10_AAL_s18
                                 + 0.00448594295768816 w_k10_AAL_s19
                                 + 0.0190077490942948 w_k10_AAL_s20
                                 + 0.0133852484337941 w_k10_AAL_s21
                                 + 0.0103410435213656 w_k10_AAL_s22
                                 + 0.0320646671958666 w_k10_AAL_s23
                                 + 0.00860527179243987 w_k10_AAL_s24
                                 + 0.0162424808154438 w_k10_AAL_s25
                                 + 0.00917892119600158 w_k10_AAL_s26
                                 + 0.0158359715307441 w_k10_AAL_s27
                                 + 0.00958391971366945 w_k10_AAL_s28
                                 + 0.0123591510229539 w_k10_AAL_s29
                                 + 0.00838468664258514 w_k10_AAL_s30
                                 + 0.00563471094266503 w_k10_AAL_s31
                                 + 0.0103129514405184 w_k10_AAL_s32
                                 + 0.0233281310376132 w_k10_AAL_s33
                                 + 0.00459929446321106 w_k10_AAL_s34
                                 + 0.00552156815007791 w_k10_AAL_s35
                                 + 0.00509393204118525 w_k10_AAL_s36
                                 + 0.00800054094228082 w_k10_AAL_s37
                                 + 0.0058700595748326 w_k10_AAL_s38
                                 + 0.00801465304389195 w_k10_AAL_s39
                                 + 0.010647275173151 w_k10_AAL_s40
                                 + 0.00647629968817428 w_k10_AAL_s41
                                 + 0.00408218243090882 w_k10_AAL_s42
                                 + 0.00907786849622712 w_k10_AAL_s43
                                 + 0.00570819312031579 w_k10_AAL_s44
                                 + 0.0104035702534231 w_k10_AAL_s45
                                 + 0.0115233056487696 w_k10_AAL_s46
                                 + 0.00540004663006152 w_k10_AAL_s47
                                 + 0.0126624552923232 w_k10_AAL_s48
                                 + 0.00427740767491994 w_k10_AAL_s49
                                 + 0.0073596870569765 w_k10_AAL_s50
                                 + 0.00253942012884562 w_k10_AAL_s51
                                 + 0.00249860033378192 w_k10_AAL_s52
                                 + 0.00375319454400121 w_k10_AAL_s53
                                 + 0.00287737498216497 w_k10_AAL_s54
                                 + 0.00714041662394175 w_k10_AAL_s55
                                 + 0.00318057817264811 w_k10_AAL_s56
                                 + 0.00474414459105258 w_k10_AAL_s57
                                 + 0.00532643281121777 w_k10_AAL_s58
                                 + 0.00448136388117054 w_k10_AAL_s59
                                 + 0.0043820316402593 w_k10_AAL_s60
                                 + 0.00612432627974316 w_k10_AAL_s61
                                 + 0.0049097927096893 w_k10_AAL_s62
                                 + 0.012429340383374 w_k10_AAL_s63
                                 + 0.00360471140013978 w_k10_AAL_s64
                                 + 0.0141797224551588 w_k10_AAL_s65
                                 + 0.0059466623255532 w_k10_AAL_s66
                                 + 0.00453221729588765 w_k10_AAL_s67
                                 + 0.0111887743849156 w_k10_AAL_s68
                                 + 0.00477706649807463 w_k10_AAL_s69
                                 + 0.00731314666355815 w_k10_AAL_s70
                                 + 0.00728517763094177 w_k10_AAL_s71
                                 + 0.00987369153196568 w_k10_AAL_s72
                                 + 0.00370051159766574 w_k10_AAL_s73
                                 + 0.00685825031301784 w_k10_AAL_s74
                                 + 0.00516045765297294 w_k10_AAL_s75
                                 + 0.0110383038093875 w_k10_AAL_s76
                                 + 0.00476811640231713 w_k10_AAL_s77
                                 + 0.014682260771747 w_k10_AAL_s78
                                 + 0.0077204583342973 w_k10_AAL_s79
                                 + 0.0205095925677563 w_k10_AAL_s80
                                 + 0.0452683046915529 w_k10_AAL_s81
                                 + 0.011668402874964 w_k10_AAL_s82
                                 + 0.0138013706675287 w_k10_AAL_s83
                                 + 0.0112131839263324 w_k10_AAL_s84
                                 + 0.0194559855285077 w_k10_AAL_s85
                                 + 0.0118187855798694 w_k10_AAL_s86
                                 + 0.0106823233259809 w_k10_AAL_s87
                                 + 0.0207375560054899 w_k10_AAL_s88
                                 + 0.0111425520220097 w_k10_AAL_s89
                                 + 0.0114517436951864 w_k10_AAL_s90
                                 + 0.00954595414330818 w_k10_AAL_s91
                                 + 0.00448302489827372 w_k10_AAL_s92
                                 + 0.00308974802506203 w_k10_AAL_s93
                                 + 0.00737654409886931 w_k10_AAL_s94
                                 + 0.0144239998431512 w_k10_AAL_s95
                                 + 0.00962806622567743 w_k10_AAL_s96
                                 + 0.0153636579008621 w_k10_AAL_s97
                                 + 0.0117468659281919 w_k10_AAL_s98
                                 + 0.00544674079362034 w_k10_AAL_s99
                                 + 0.0335286170248049 w_k10_PRU_s00
                                 + 0.0108103677641175 w_k10_PRU_s01
                                 + 0.0114755186142514 w_k10_PRU_s02
                                 + 0.0117636167549145 w_k10_PRU_s03
                                 + 0.00859075389635551 w_k10_PRU_s04
                                 + 0.0130565716495212 w_k10_PRU_s05
                                 + 0.0163618207725818 w_k10_PRU_s06
                                 + 0.00852875961326144 w_k10_PRU_s07
                                 + 0.00693898714982084 w_k10_PRU_s08
                                 + 0.0142030642585276 w_k10_PRU_s09
                                 + 0.011574653911378 w_k10_PRU_s10
                                 + 0.00657474381512374 w_k10_PRU_s11
                                 + 0.0148108604070829 w_k10_PRU_s12
                                 + 0.0062918877093065 w_k10_PRU_s13
                                 + 0.0179773649038015 w_k10_PRU_s14
                                 + 0.00841558619641052 w_k10_PRU_s15
                                 + 0.00563013662409166 w_k10_PRU_s16
                                 + 0.0104134604100603 w_k10_PRU_s17
                                 + 0.00846461852496519 w_k10_PRU_s18
                                 + 0.00446365969384172 w_k10_PRU_s19
                                 + 0.0181324994760761 w_k10_PRU_s20
                                 + 0.0131996555868546 w_k10_PRU_s21
                                 + 0.0102621432652059 w_k10_PRU_s22
                                 + 0.032911063173893 w_k10_PRU_s23
                                 + 0.00767461086290813 w_k10_PRU_s24
                                 + 0.0173393840331908 w_k10_PRU_s25
                                 + 0.009747688320189 w_k10_PRU_s26
                                 + 0.0190505255450252 w_k10_PRU_s27
                                 + 0.00850146348356951 w_k10_PRU_s28
                                 + 0.0116029866503993 w_k10_PRU_s29
                                 + 0.00701842338321833 w_k10_PRU_s30
                                 + 0.00587054545502163 w_k10_PRU_s31
                                 + 0.00968826182168626 w_k10_PRU_s32
                                 + 0.0219746968057066 w_k10_PRU_s33
                                 + 0.00499525932580723 w_k10_PRU_s34
                                 + 0.00585388166995306 w_k10_PRU_s35
                                 + 0.00485920965338891 w_k10_PRU_s36
                                 + 0.00846571882031799 w_k10_PRU_s37
                                 + 0.00541671797118789 w_k10_PRU_s38
                                 + 0.00826424008286456 w_k10_PRU_s39
                                 + 0.0112600405657719 w_k10_PRU_s40
                                 + 0.00706061998093763 w_k10_PRU_s41
                                 + 0.00593676745987625 w_k10_PRU_s42
                                 + 0.00877385506167746 w_k10_PRU_s43
                                 + 0.00563538925360057 w_k10_PRU_s44
                                 + 0.0100123070788387 w_k10_PRU_s45
                                 + 0.0101845422180225 w_k10_PRU_s46
                                 + 0.00510249658842108 w_k10_PRU_s47
                                 + 0.0124074214680718 w_k10_PRU_s48
                                 + 0.00370988328956127 w_k10_PRU_s49
                                 + 0.00659256951408775 w_k10_PRU_s50
                                 + 0.0025815998012033 w_k10_PRU_s51
                                 + 0.00231655830414717 w_k10_PRU_s52
                                 + 0.00380617897074842 w_k10_PRU_s53
                                 + 0.00282943438414284 w_k10_PRU_s54
                                 + 0.0075266753251033 w_k10_PRU_s55
                                 + 0.00289714633246057 w_k10_PRU_s56
                                 + 0.00489975481975135 w_k10_PRU_s57
                                 + 0.00519541078237395 w_k10_PRU_s58
                                 + 0.00465521862694892 w_k10_PRU_s59
                                 + 0.00451303130416095 w_k10_PRU_s60
                                 + 0.00760101960610699 w_k10_PRU_s61
                                 + 0.00523854411011276 w_k10_PRU_s62
                                 + 0.0148238993402265 w_k10_PRU_s63
                                 + 0.00420612212339991 w_k10_PRU_s64
                                 + 0.015086063244484 w_k10_PRU_s65
                                 + 0.0064859732632008 w_k10_PRU_s66
                                 + 0.00455046625501758 w_k10_PRU_s67
                                 + 0.00957249711554482 w_k10_PRU_s68
                                 + 0.0049182658955245 w_k10_PRU_s69
                                 + 0.0069718457900602 w_k10_PRU_s70
                                 + 0.00682093070451348 w_k10_PRU_s71
                                 + 0.0098061825913035 w_k10_PRU_s72
                                 + 0.00408968131885324 w_k10_PRU_s73
                                 + 0.00650576433977903 w_k10_PRU_s74
                                 + 0.00522050892745441 w_k10_PRU_s75
                                 + 0.0107929933620383 w_k10_PRU_s76
                                 + 0.0043240107733857 w_k10_PRU_s77
                                 + 0.0156584490638952 w_k10_PRU_s78
                                 + 0.00891231218686537 w_k10_PRU_s79
                                 + 0.0205649973272987 w_k10_PRU_s80
                                 + 0.041594974642524 w_k10_PRU_s81
                                 + 0.0112453971157339 w_k10_PRU_s82
                                 + 0.01381363287893 w_k10_PRU_s83
                                 + 0.0112074567373932 w_k10_PRU_s84
                                 + 0.0188174873820678 w_k10_PRU_s85
                                 + 0.0130687416316632 w_k10_PRU_s86
                                 + 0.0101838971152608 w_k10_PRU_s87
                                 + 0.0220945274325096 w_k10_PRU_s88
                                 + 0.00968896608576118 w_k10_PRU_s89
                                 + 0.00913579740601767 w_k10_PRU_s90
                                 + 0.00996580017460652 w_k10_PRU_s91
                                 + 0.00502778637546179 w_k10_PRU_s92
                                 + 0.00310225662040992 w_k10_PRU_s93
                                 + 0.00668777766307719 w_k10_PRU_s94
                                 + 0.0153217313861774 w_k10_PRU_s95
                                 + 0.00977325519691169 w_k10_PRU_s96
                                 + 0.0171442053545597 w_k10_PRU_s97
                                 + 0.0114627954240219 w_k10_PRU_s98
                                 + 0.00546878407227721 w_k10_PRU_s99 >= 1.08
Bounds
 0 <= w_0_HSBC <= 1
 0 <= w_0_VOD <= 1
 0 <= w_0_BP <= 1
 0 <= w_0_GSK <= 1
 0 <= w_0_AZN <= 1
 0 <= w_0_RIO <= 1
 0 <= w_0_BG <= 1
 0 <= w_0_TSCO <= 1
 0 <= w_0_AAL <= 1
 0 <= w_0_PRU <= 1
 0 <= w_k10_HSBC_s0 <= 1
 0 <= w_k10_HSBC_s1 <= 1
 0 <= w_k10_HSBC_s2 <= 1
 0 <= w_k10_HSBC_s3 <= 1
 0 <= w_k10_HSBC_s4 <= 1
 0 <= w_k10_HSBC_s5 <= 1
 0 <= w_k10_HSBC_s6 <= 1
 0 <= w_k10_HSBC_s7 <= 1
 0 <= w_k10_HSBC_s8 <= 1
 0 <= w_k10_HSBC_s9 <= 1
 0 <= w_k10_HSBC_s00 <= 1
 0 <= w_k10_HSBC_s01 <= 1
 0 <= w_k10_HSBC_s02 <= 1
 0 <= w_k10_HSBC_s03 <= 1
 0 <= w_k10_HSBC_s04 <= 1
 0 <= w_k10_HSBC_s05 <= 1
 0 <= w_k10_HSBC_s06 <= 1
 0 <= w_k10_HSBC_s07 <= 1
 0 <= w_k10_HSBC_s08 <= 1
 0 <= w_k10_HSBC_s09 <= 1
 0 <= w_k10_HSBC_s10 <= 1
 0 <= w_k10_HSBC_s11 <= 1
 0 <= w_k10_HSBC_s12 <= 1
 0 <= w_k10_HSBC_s13 <= 1
 0 <= w_k10_HSBC_s14 <= 1
 0 <= w_k10_HSBC_s15 <= 1
 0 <= w_k10_HSBC_s16 <= 1
 0 <= w_k10_HSBC_s17 <= 1
 0 <= w_k10_HSBC_s18 <= 1
 0 <= w_k10_HSBC_s19 <= 1
 0 <= w_k10_HSBC_s20 <= 1
 0 <= w_k10_HSBC_s21 <= 1
 0 <= w_k10_HSBC_s22 <= 1
 0 <= w_k10_HSBC_s23 <= 1
 0 <= w_k10_HSBC_s24 <= 1
 0 <= w_k10_HSBC_s25 <= 1
 0 <= w_k10_HSBC_s26 <= 1
 0 <= w_k10_HSBC_s27 <= 1
 0 <= w_k10_HSBC_s28 <= 1
 0 <= w_k10_HSBC_s29 <= 1
 0 <= w_k10_HSBC_s30 <= 1
 0 <= w_k10_HSBC_s31 <= 1
 0 <= w_k10_HSBC_s32 <= 1
 0 <= w_k10_HSBC_s33 <= 1
 0 <= w_k10_HSBC_s34 <= 1
 0 <= w_k10_HSBC_s35 <= 1
 0 <= w_k10_HSBC_s36 <= 1
 0 <= w_k10_HSBC_s37 <= 1
 0 <= w_k10_HSBC_s38 <= 1
 0 <= w_k10_HSBC_s39 <= 1
 0 <= w_k10_HSBC_s40 <= 1
 0 <= w_k10_HSBC_s41 <= 1
 0 <= w_k10_HSBC_s42 <= 1
 0 <= w_k10_HSBC_s43 <= 1
 0 <= w_k10_HSBC_s44 <= 1
 0 <= w_k10_HSBC_s45 <= 1
 0 <= w_k10_HSBC_s46 <= 1
 0 <= w_k10_HSBC_s47 <= 1
 0 <= w_k10_HSBC_s48 <= 1
 0 <= w_k10_HSBC_s49 <= 1
 0 <= w_k10_HSBC_s50 <= 1
 0 <= w_k10_HSBC_s51 <= 1
 0 <= w_k10_HSBC_s52 <= 1
 0 <= w_k10_HSBC_s53 <= 1
 0 <= w_k10_HSBC_s54 <= 1
 0 <= w_k10_HSBC_s55 <= 1
 0 <= w_k10_HSBC_s56 <= 1
 0 <= w_k10_HSBC_s57 <= 1
 0 <= w_k10_HSBC_s58 <= 1
 0 <= w_k10_HSBC_s59 <= 1
 0 <= w_k10_HSBC_s60 <= 1
 0 <= w_k10_HSBC_s61 <= 1
 0 <= w_k10_HSBC_s62 <= 1
 0 <= w_k10_HSBC_s63 <= 1
 0 <= w_k10_HSBC_s64 <= 1
 0 <= w_k10_HSBC_s65 <= 1
 0 <= w_k10_HSBC_s66 <= 1
 0 <= w_k10_HSBC_s67 <= 1
 0 <= w_k10_HSBC_s68 <= 1
 0 <= w_k10_HSBC_s69 <= 1
 0 <= w_k10_HSBC_s70 <= 1
 0 <= w_k10_HSBC_s71 <= 1
 0 <= w_k10_HSBC_s72 <= 1
 0 <= w_k10_HSBC_s73 <= 1
 0 <= w_k10_HSBC_s74 <= 1
 0 <= w_k10_HSBC_s75 <= 1
 0 <= w_k10_HSBC_s76 <= 1
 0 <= w_k10_HSBC_s77 <= 1
 0 <= w_k10_HSBC_s78 <= 1
 0 <= w_k10_HSBC_s79 <= 1
 0 <= w_k10_HSBC_s80 <= 1
 0 <= w_k10_HSBC_s81 <= 1
 0 <= w_k10_HSBC_s82 <= 1
 0 <= w_k10_HSBC_s83 <= 1
 0 <= w_k10_HSBC_s84 <= 1
 0 <= w_k10_HSBC_s85 <= 1
 0 <= w_k10_HSBC_s86 <= 1
 0 <= w_k10_HSBC_s87 <= 1
 0 <= w_k10_HSBC_s88 <= 1
 0 <= w_k10_HSBC_s89 <= 1
 0 <= w_k10_HSBC_s90 <= 1
 0 <= w_k10_HSBC_s91 <= 1
 0 <= w_k10_HSBC_s92 <= 1
 0 <= w_k10_HSBC_s93 <= 1
 0 <= w_k10_HSBC_s94 <= 1
 0 <= w_k10_HSBC_s95 <= 1
 0 <= w_k10_HSBC_s96 <= 1
 0 <= w_k10_HSBC_s97 <= 1
 0 <= w_k10_HSBC_s98 <= 1
 0 <= w_k10_HSBC_s99 <= 1
 0 <= w_k10_VOD_s0 <= 1
 0 <= w_k10_VOD_s1 <= 1
 0 <= w_k10_VOD_s2 <= 1
 0 <= w_k10_VOD_s3 <= 1
 0 <= w_k10_VOD_s4 <= 1
 0 <= w_k10_VOD_s5 <= 1
 0 <= w_k10_VOD_s6 <= 1
 0 <= w_k10_VOD_s7 <= 1
 0 <= w_k10_VOD_s8 <= 1
 0 <= w_k10_VOD_s9 <= 1
 0 <= w_k10_VOD_s00 <= 1
 0 <= w_k10_VOD_s01 <= 1
 0 <= w_k10_VOD_s02 <= 1
 0 <= w_k10_VOD_s03 <= 1
 0 <= w_k10_VOD_s04 <= 1
 0 <= w_k10_VOD_s05 <= 1
 0 <= w_k10_VOD_s06 <= 1
 0 <= w_k10_VOD_s07 <= 1
 0 <= w_k10_VOD_s08 <= 1
 0 <= w_k10_VOD_s09 <= 1
 0 <= w_k10_VOD_s10 <= 1
 0 <= w_k10_VOD_s11 <= 1
 0 <= w_k10_VOD_s12 <= 1
 0 <= w_k10_VOD_s13 <= 1
 0 <= w_k10_VOD_s14 <= 1
 0 <= w_k10_VOD_s15 <= 1
 0 <= w_k10_VOD_s16 <= 1
 0 <= w_k10_VOD_s17 <= 1
 0 <= w_k10_VOD_s18 <= 1
 0 <= w_k10_VOD_s19 <= 1
 0 <= w_k10_VOD_s20 <= 1
 0 <= w_k10_VOD_s21 <= 1
 0 <= w_k10_VOD_s22 <= 1
 0 <= w_k10_VOD_s23 <= 1
 0 <= w_k10_VOD_s24 <= 1
 0 <= w_k10_VOD_s25 <= 1
 0 <= w_k10_VOD_s26 <= 1
 0 <= w_k10_VOD_s27 <= 1
 0 <= w_k10_VOD_s28 <= 1
 0 <= w_k10_VOD_s29 <= 1
 0 <= w_k10_VOD_s30 <= 1
 0 <= w_k10_VOD_s31 <= 1
 0 <= w_k10_VOD_s32 <= 1
 0 <= w_k10_VOD_s33 <= 1
 0 <= w_k10_VOD_s34 <= 1
 0 <= w_k10_VOD_s35 <= 1
 0 <= w_k10_VOD_s36 <= 1
 0 <= w_k10_VOD_s37 <= 1
 0 <= w_k10_VOD_s38 <= 1
 0 <= w_k10_VOD_s39 <= 1
 0 <= w_k10_VOD_s40 <= 1
 0 <= w_k10_VOD_s41 <= 1
 0 <= w_k10_VOD_s42 <= 1
 0 <= w_k10_VOD_s43 <= 1
 0 <= w_k10_VOD_s44 <= 1
 0 <= w_k10_VOD_s45 <= 1
 0 <= w_k10_VOD_s46 <= 1
 0 <= w_k10_VOD_s47 <= 1
 0 <= w_k10_VOD_s48 <= 1
 0 <= w_k10_VOD_s49 <= 1
 0 <= w_k10_VOD_s50 <= 1
 0 <= w_k10_VOD_s51 <= 1
 0 <= w_k10_VOD_s52 <= 1
 0 <= w_k10_VOD_s53 <= 1
 0 <= w_k10_VOD_s54 <= 1
 0 <= w_k10_VOD_s55 <= 1
 0 <= w_k10_VOD_s56 <= 1
 0 <= w_k10_VOD_s57 <= 1
 0 <= w_k10_VOD_s58 <= 1
 0 <= w_k10_VOD_s59 <= 1
 0 <= w_k10_VOD_s60 <= 1
 0 <= w_k10_VOD_s61 <= 1
 0 <= w_k10_VOD_s62 <= 1
 0 <= w_k10_VOD_s63 <= 1
 0 <= w_k10_VOD_s64 <= 1
 0 <= w_k10_VOD_s65 <= 1
 0 <= w_k10_VOD_s66 <= 1
 0 <= w_k10_VOD_s67 <= 1
 0 <= w_k10_VOD_s68 <= 1
 0 <= w_k10_VOD_s69 <= 1
 0 <= w_k10_VOD_s70 <= 1
 0 <= w_k10_VOD_s71 <= 1
 0 <= w_k10_VOD_s72 <= 1
 0 <= w_k10_VOD_s73 <= 1
 0 <= w_k10_VOD_s74 <= 1
 0 <= w_k10_VOD_s75 <= 1
 0 <= w_k10_VOD_s76 <= 1
 0 <= w_k10_VOD_s77 <= 1
 0 <= w_k10_VOD_s78 <= 1
 0 <= w_k10_VOD_s79 <= 1
 0 <= w_k10_VOD_s80 <= 1
 0 <= w_k10_VOD_s81 <= 1
 0 <= w_k10_VOD_s82 <= 1
 0 <= w_k10_VOD_s83 <= 1
 0 <= w_k10_VOD_s84 <= 1
 0 <= w_k10_VOD_s85 <= 1
 0 <= w_k10_VOD_s86 <= 1
 0 <= w_k10_VOD_s87 <= 1
 0 <= w_k10_VOD_s88 <= 1
 0 <= w_k10_VOD_s89 <= 1
 0 <= w_k10_VOD_s90 <= 1
 0 <= w_k10_VOD_s91 <= 1
 0 <= w_k10_VOD_s92 <= 1
 0 <= w_k10_VOD_s93 <= 1
 0 <= w_k10_VOD_s94 <= 1
 0 <= w_k10_VOD_s95 <= 1
 0 <= w_k10_VOD_s96 <= 1
 0 <= w_k10_VOD_s97 <= 1
 0 <= w_k10_VOD_s98 <= 1
 0 <= w_k10_VOD_s99 <= 1
 0 <= w_k10_BP_s0 <= 1
 0 <= w_k10_BP_s1 <= 1
 0 <= w_k10_BP_s2 <= 1
 0 <= w_k10_BP_s3 <= 1
 0 <= w_k10_BP_s4 <= 1
 0 <= w_k10_BP_s5 <= 1
 0 <= w_k10_BP_s6 <= 1
 0 <= w_k10_BP_s7 <= 1
 0 <= w_k10_BP_s8 <= 1
 0 <= w_k10_BP_s9 <= 1
 0 <= w_k10_BP_s00 <= 1
 0 <= w_k10_BP_s01 <= 1
 0 <= w_k10_BP_s02 <= 1
 0 <= w_k10_BP_s03 <= 1
 0 <= w_k10_BP_s04 <= 1
 0 <= w_k10_BP_s05 <= 1
 0 <= w_k10_BP_s06 <= 1
 0 <= w_k10_BP_s07 <= 1
 0 <= w_k10_BP_s08 <= 1
 0 <= w_k10_BP_s09 <= 1
 0 <= w_k10_BP_s10 <= 1
 0 <= w_k10_BP_s11 <= 1
 0 <= w_k10_BP_s12 <= 1
 0 <= w_k10_BP_s13 <= 1
 0 <= w_k10_BP_s14 <= 1
 0 <= w_k10_BP_s15 <= 1
 0 <= w_k10_BP_s16 <= 1
 0 <= w_k10_BP_s17 <= 1
 0 <= w_k10_BP_s18 <= 1
 0 <= w_k10_BP_s19 <= 1
 0 <= w_k10_BP_s20 <= 1
 0 <= w_k10_BP_s21 <= 1
 0 <= w_k10_BP_s22 <= 1
 0 <= w_k10_BP_s23 <= 1
 0 <= w_k10_BP_s24 <= 1
 0 <= w_k10_BP_s25 <= 1
 0 <= w_k10_BP_s26 <= 1
 0 <= w_k10_BP_s27 <= 1
 0 <= w_k10_BP_s28 <= 1
 0 <= w_k10_BP_s29 <= 1
 0 <= w_k10_BP_s30 <= 1
 0 <= w_k10_BP_s31 <= 1
 0 <= w_k10_BP_s32 <= 1
 0 <= w_k10_BP_s33 <= 1
 0 <= w_k10_BP_s34 <= 1
 0 <= w_k10_BP_s35 <= 1
 0 <= w_k10_BP_s36 <= 1
 0 <= w_k10_BP_s37 <= 1
 0 <= w_k10_BP_s38 <= 1
 0 <= w_k10_BP_s39 <= 1
 0 <= w_k10_BP_s40 <= 1
 0 <= w_k10_BP_s41 <= 1
 0 <= w_k10_BP_s42 <= 1
 0 <= w_k10_BP_s43 <= 1
 0 <= w_k10_BP_s44 <= 1
 0 <= w_k10_BP_s45 <= 1
 0 <= w_k10_BP_s46 <= 1
 0 <= w_k10_BP_s47 <= 1
 0 <= w_k10_BP_s48 <= 1
 0 <= w_k10_BP_s49 <= 1
 0 <= w_k10_BP_s50 <= 1
 0 <= w_k10_BP_s51 <= 1
 0 <= w_k10_BP_s52 <= 1
 0 <= w_k10_BP_s53 <= 1
 0 <= w_k10_BP_s54 <= 1
 0 <= w_k10_BP_s55 <= 1
 0 <= w_k10_BP_s56 <= 1
 0 <= w_k10_BP_s57 <= 1
 0 <= w_k10_BP_s58 <= 1
 0 <= w_k10_BP_s59 <= 1
 0 <= w_k10_BP_s60 <= 1
 0 <= w_k10_BP_s61 <= 1
 0 <= w_k10_BP_s62 <= 1
 0 <= w_k10_BP_s63 <= 1
 0 <= w_k10_BP_s64 <= 1
 0 <= w_k10_BP_s65 <= 1
 0 <= w_k10_BP_s66 <= 1
 0 <= w_k10_BP_s67 <= 1
 0 <= w_k10_BP_s68 <= 1
 0 <= w_k10_BP_s69 <= 1
 0 <= w_k10_BP_s70 <= 1
 0 <= w_k10_BP_s71 <= 1
 0 <= w_k10_BP_s72 <= 1
 0 <= w_k10_BP_s73 <= 1
 0 <= w_k10_BP_s74 <= 1
 0 <= w_k10_BP_s75 <= 1
 0 <= w_k10_BP_s76 <= 1
 0 <= w_k10_BP_s77 <= 1
 0 <= w_k10_BP_s78 <= 1
 0 <= w_k10_BP_s79 <= 1
 0 <= w_k10_BP_s80 <= 1
 0 <= w_k10_BP_s81 <= 1
 0 <= w_k10_BP_s82 <= 1
 0 <= w_k10_BP_s83 <= 1
 0 <= w_k10_BP_s84 <= 1
 0 <= w_k10_BP_s85 <= 1
 0 <= w_k10_BP_s86 <= 1
 0 <= w_k10_BP_s87 <= 1
 0 <= w_k10_BP_s88 <= 1
 0 <= w_k10_BP_s89 <= 1
 0 <= w_k10_BP_s90 <= 1
 0 <= w_k10_BP_s91 <= 1
 0 <= w_k10_BP_s92 <= 1
 0 <= w_k10_BP_s93 <= 1
 0 <= w_k10_BP_s94 <= 1
 0 <= w_k10_BP_s95 <= 1
 0 <= w_k10_BP_s96 <= 1
 0 <= w_k10_BP_s97 <= 1
 0 <= w_k10_BP_s98 <= 1
 0 <= w_k10_BP_s99 <= 1
 0 <= w_k10_GSK_s0 <= 1
 0 <= w_k10_GSK_s1 <= 1
 0 <= w_k10_GSK_s2 <= 1
 0 <= w_k10_GSK_s3 <= 1
 0 <= w_k10_GSK_s4 <= 1
 0 <= w_k10_GSK_s5 <= 1
 0 <= w_k10_GSK_s6 <= 1
 0 <= w_k10_GSK_s7 <= 1
 0 <= w_k10_GSK_s8 <= 1
 0 <= w_k10_GSK_s9 <= 1
 0 <= w_k10_GSK_s00 <= 1
 0 <= w_k10_GSK_s01 <= 1
 0 <= w_k10_GSK_s02 <= 1
 0 <= w_k10_GSK_s03 <= 1
 0 <= w_k10_GSK_s04 <= 1
 0 <= w_k10_GSK_s05 <= 1
 0 <= w_k10_GSK_s06 <= 1
 0 <= w_k10_GSK_s07 <= 1
 0 <= w_k10_GSK_s08 <= 1
 0 <= w_k10_GSK_s09 <= 1
 0 <= w_k10_GSK_s10 <= 1
 0 <= w_k10_GSK_s11 <= 1
 0 <= w_k10_GSK_s12 <= 1
 0 <= w_k10_GSK_s13 <= 1
 0 <= w_k10_GSK_s14 <= 1
 0 <= w_k10_GSK_s15 <= 1
 0 <= w_k10_GSK_s16 <= 1
 0 <= w_k10_GSK_s17 <= 1
 0 <= w_k10_GSK_s18 <= 1
 0 <= w_k10_GSK_s19 <= 1
 0 <= w_k10_GSK_s20 <= 1
 0 <= w_k10_GSK_s21 <= 1
 0 <= w_k10_GSK_s22 <= 1
 0 <= w_k10_GSK_s23 <= 1
 0 <= w_k10_GSK_s24 <= 1
 0 <= w_k10_GSK_s25 <= 1
 0 <= w_k10_GSK_s26 <= 1
 0 <= w_k10_GSK_s27 <= 1
 0 <= w_k10_GSK_s28 <= 1
 0 <= w_k10_GSK_s29 <= 1
 0 <= w_k10_GSK_s30 <= 1
 0 <= w_k10_GSK_s31 <= 1
 0 <= w_k10_GSK_s32 <= 1
 0 <= w_k10_GSK_s33 <= 1
 0 <= w_k10_GSK_s34 <= 1
 0 <= w_k10_GSK_s35 <= 1
 0 <= w_k10_GSK_s36 <= 1
 0 <= w_k10_GSK_s37 <= 1
 0 <= w_k10_GSK_s38 <= 1
 0 <= w_k10_GSK_s39 <= 1
 0 <= w_k10_GSK_s40 <= 1
 0 <= w_k10_GSK_s41 <= 1
 0 <= w_k10_GSK_s42 <= 1
 0 <= w_k10_GSK_s43 <= 1
 0 <= w_k10_GSK_s44 <= 1
 0 <= w_k10_GSK_s45 <= 1
 0 <= w_k10_GSK_s46 <= 1
 0 <= w_k10_GSK_s47 <= 1
 0 <= w_k10_GSK_s48 <= 1
 0 <= w_k10_GSK_s49 <= 1
 0 <= w_k10_GSK_s50 <= 1
 0 <= w_k10_GSK_s51 <= 1
 0 <= w_k10_GSK_s52 <= 1
 0 <= w_k10_GSK_s53 <= 1
 0 <= w_k10_GSK_s54 <= 1
 0 <= w_k10_GSK_s55 <= 1
 0 <= w_k10_GSK_s56 <= 1
 0 <= w_k10_GSK_s57 <= 1
 0 <= w_k10_GSK_s58 <= 1
 0 <= w_k10_GSK_s59 <= 1
 0 <= w_k10_GSK_s60 <= 1
 0 <= w_k10_GSK_s61 <= 1
 0 <= w_k10_GSK_s62 <= 1
 0 <= w_k10_GSK_s63 <= 1
 0 <= w_k10_GSK_s64 <= 1
 0 <= w_k10_GSK_s65 <= 1
 0 <= w_k10_GSK_s66 <= 1
 0 <= w_k10_GSK_s67 <= 1
 0 <= w_k10_GSK_s68 <= 1
 0 <= w_k10_GSK_s69 <= 1
 0 <= w_k10_GSK_s70 <= 1
 0 <= w_k10_GSK_s71 <= 1
 0 <= w_k10_GSK_s72 <= 1
 0 <= w_k10_GSK_s73 <= 1
 0 <= w_k10_GSK_s74 <= 1
 0 <= w_k10_GSK_s75 <= 1
 0 <= w_k10_GSK_s76 <= 1
 0 <= w_k10_GSK_s77 <= 1
 0 <= w_k10_GSK_s78 <= 1
 0 <= w_k10_GSK_s79 <= 1
 0 <= w_k10_GSK_s80 <= 1
 0 <= w_k10_GSK_s81 <= 1
 0 <= w_k10_GSK_s82 <= 1
 0 <= w_k10_GSK_s83 <= 1
 0 <= w_k10_GSK_s84 <= 1
 0 <= w_k10_GSK_s85 <= 1
 0 <= w_k10_GSK_s86 <= 1
 0 <= w_k10_GSK_s87 <= 1
 0 <= w_k10_GSK_s88 <= 1
 0 <= w_k10_GSK_s89 <= 1
 0 <= w_k10_GSK_s90 <= 1
 0 <= w_k10_GSK_s91 <= 1
 0 <= w_k10_GSK_s92 <= 1
 0 <= w_k10_GSK_s93 <= 1
 0 <= w_k10_GSK_s94 <= 1
 0 <= w_k10_GSK_s95 <= 1
 0 <= w_k10_GSK_s96 <= 1
 0 <= w_k10_GSK_s97 <= 1
 0 <= w_k10_GSK_s98 <= 1
 0 <= w_k10_GSK_s99 <= 1
 0 <= w_k10_AZN_s0 <= 1
 0 <= w_k10_AZN_s1 <= 1
 0 <= w_k10_AZN_s2 <= 1
 0 <= w_k10_AZN_s3 <= 1
 0 <= w_k10_AZN_s4 <= 1
 0 <= w_k10_AZN_s5 <= 1
 0 <= w_k10_AZN_s6 <= 1
 0 <= w_k10_AZN_s7 <= 1
 0 <= w_k10_AZN_s8 <= 1
 0 <= w_k10_AZN_s9 <= 1
 0 <= w_k10_AZN_s00 <= 1
 0 <= w_k10_AZN_s01 <= 1
 0 <= w_k10_AZN_s02 <= 1
 0 <= w_k10_AZN_s03 <= 1
 0 <= w_k10_AZN_s04 <= 1
 0 <= w_k10_AZN_s05 <= 1
 0 <= w_k10_AZN_s06 <= 1
 0 <= w_k10_AZN_s07 <= 1
 0 <= w_k10_AZN_s08 <= 1
 0 <= w_k10_AZN_s09 <= 1
 0 <= w_k10_AZN_s10 <= 1
 0 <= w_k10_AZN_s11 <= 1
 0 <= w_k10_AZN_s12 <= 1
 0 <= w_k10_AZN_s13 <= 1
 0 <= w_k10_AZN_s14 <= 1
 0 <= w_k10_AZN_s15 <= 1
 0 <= w_k10_AZN_s16 <= 1
 0 <= w_k10_AZN_s17 <= 1
 0 <= w_k10_AZN_s18 <= 1
 0 <= w_k10_AZN_s19 <= 1
 0 <= w_k10_AZN_s20 <= 1
 0 <= w_k10_AZN_s21 <= 1
 0 <= w_k10_AZN_s22 <= 1
 0 <= w_k10_AZN_s23 <= 1
 0 <= w_k10_AZN_s24 <= 1
 0 <= w_k10_AZN_s25 <= 1
 0 <= w_k10_AZN_s26 <= 1
 0 <= w_k10_AZN_s27 <= 1
 0 <= w_k10_AZN_s28 <= 1
 0 <= w_k10_AZN_s29 <= 1
 0 <= w_k10_AZN_s30 <= 1
 0 <= w_k10_AZN_s31 <= 1
 0 <= w_k10_AZN_s32 <= 1
 0 <= w_k10_AZN_s33 <= 1
 0 <= w_k10_AZN_s34 <= 1
 0 <= w_k10_AZN_s35 <= 1
 0 <= w_k10_AZN_s36 <= 1
 0 <= w_k10_AZN_s37 <= 1
 0 <= w_k10_AZN_s38 <= 1
 0 <= w_k10_AZN_s39 <= 1
 0 <= w_k10_AZN_s40 <= 1
 0 <= w_k10_AZN_s41 <= 1
 0 <= w_k10_AZN_s42 <= 1
 0 <= w_k10_AZN_s43 <= 1
 0 <= w_k10_AZN_s44 <= 1
 0 <= w_k10_AZN_s45 <= 1
 0 <= w_k10_AZN_s46 <= 1
 0 <= w_k10_AZN_s47 <= 1
 0 <= w_k10_AZN_s48 <= 1
 0 <= w_k10_AZN_s49 <= 1
 0 <= w_k10_AZN_s50 <= 1
 0 <= w_k10_AZN_s51 <= 1
 0 <= w_k10_AZN_s52 <= 1
 0 <= w_k10_AZN_s53 <= 1
 0 <= w_k10_AZN_s54 <= 1
 0 <= w_k10_AZN_s55 <= 1
 0 <= w_k10_AZN_s56 <= 1
 0 <= w_k10_AZN_s57 <= 1
 0 <= w_k10_AZN_s58 <= 1
 0 <= w_k10_AZN_s59 <= 1
 0 <= w_k10_AZN_s60 <= 1
 0 <= w_k10_AZN_s61 <= 1
 0 <= w_k10_AZN_s62 <= 1
 0 <= w_k10_AZN_s63 <= 1
 0 <= w_k10_AZN_s64 <= 1
 0 <= w_k10_AZN_s65 <= 1
 0 <= w_k10_AZN_s66 <= 1
 0 <= w_k10_AZN_s67 <= 1
 0 <= w_k10_AZN_s68 <= 1
 0 <= w_k10_AZN_s69 <= 1
 0 <= w_k10_AZN_s70 <= 1
 0 <= w_k10_AZN_s71 <= 1
 0 <= w_k10_AZN_s72 <= 1
 0 <= w_k10_AZN_s73 <= 1
 0 <= w_k10_AZN_s74 <= 1
 0 <= w_k10_AZN_s75 <= 1
 0 <= w_k10_AZN_s76 <= 1
 0 <= w_k10_AZN_s77 <= 1
 0 <= w_k10_AZN_s78 <= 1
 0 <= w_k10_AZN_s79 <= 1
 0 <= w_k10_AZN_s80 <= 1
 0 <= w_k10_AZN_s81 <= 1
 0 <= w_k10_AZN_s82 <= 1
 0 <= w_k10_AZN_s83 <= 1
 0 <= w_k10_AZN_s84 <= 1
 0 <= w_k10_AZN_s85 <= 1
 0 <= w_k10_AZN_s86 <= 1
 0 <= w_k10_AZN_s87 <= 1
 0 <= w_k10_AZN_s88 <= 1
 0 <= w_k10_AZN_s89 <= 1
 0 <= w_k10_AZN_s90 <= 1
 0 <= w_k10_AZN_s91 <= 1
 0 <= w_k10_AZN_s92 <= 1
 0 <= w_k10_AZN_s93 <= 1
 0 <= w_k10_AZN_s94 <= 1
 0 <= w_k10_AZN_s95 <= 1
 0 <= w_k10_AZN_s96 <= 1
 0 <= w_k10_AZN_s97 <= 1
 0 <= w_k10_AZN_s98 <= 1
 0 <= w_k10_AZN_s99 <= 1
 0 <= w_k10_RIO_s0 <= 1
 0 <= w_k10_RIO_s1 <= 1
 0 <= w_k10_RIO_s2 <= 1
 0 <= w_k10_RIO_s3 <= 1
 0 <= w_k10_RIO_s4 <= 1
 0 <= w_k10_RIO_s5 <= 1
 0 <= w_k10_RIO_s6 <= 1
 0 <= w_k10_RIO_s7 <= 1
 0 <= w_k10_RIO_s8 <= 1
 0 <= w_k10_RIO_s9 <= 1
 0 <= w_k10_RIO_s00 <= 1
 0 <= w_k10_RIO_s01 <= 1
 0 <= w_k10_RIO_s02 <= 1
 0 <= w_k10_RIO_s03 <= 1
 0 <= w_k10_RIO_s04 <= 1
 0 <= w_k10_RIO_s05 <= 1
 0 <= w_k10_RIO_s06 <= 1
 0 <= w_k10_RIO_s07 <= 1
 0 <= w_k10_RIO_s08 <= 1
 0 <= w_k10_RIO_s09 <= 1
 0 <= w_k10_RIO_s10 <= 1
 0 <= w_k10_RIO_s11 <= 1
 0 <= w_k10_RIO_s12 <= 1
 0 <= w_k10_RIO_s13 <= 1
 0 <= w_k10_RIO_s14 <= 1
 0 <= w_k10_RIO_s15 <= 1
 0 <= w_k10_RIO_s16 <= 1
 0 <= w_k10_RIO_s17 <= 1
 0 <= w_k10_RIO_s18 <= 1
 0 <= w_k10_RIO_s19 <= 1
 0 <= w_k10_RIO_s20 <= 1
 0 <= w_k10_RIO_s21 <= 1
 0 <= w_k10_RIO_s22 <= 1
 0 <= w_k10_RIO_s23 <= 1
 0 <= w_k10_RIO_s24 <= 1
 0 <= w_k10_RIO_s25 <= 1
 0 <= w_k10_RIO_s26 <= 1
 0 <= w_k10_RIO_s27 <= 1
 0 <= w_k10_RIO_s28 <= 1
 0 <= w_k10_RIO_s29 <= 1
 0 <= w_k10_RIO_s30 <= 1
 0 <= w_k10_RIO_s31 <= 1
 0 <= w_k10_RIO_s32 <= 1
 0 <= w_k10_RIO_s33 <= 1
 0 <= w_k10_RIO_s34 <= 1
 0 <= w_k10_RIO_s35 <= 1
 0 <= w_k10_RIO_s36 <= 1
 0 <= w_k10_RIO_s37 <= 1
 0 <= w_k10_RIO_s38 <= 1
 0 <= w_k10_RIO_s39 <= 1
 0 <= w_k10_RIO_s40 <= 1
 0 <= w_k10_RIO_s41 <= 1
 0 <= w_k10_RIO_s42 <= 1
 0 <= w_k10_RIO_s43 <= 1
 0 <= w_k10_RIO_s44 <= 1
 0 <= w_k10_RIO_s45 <= 1
 0 <= w_k10_RIO_s46 <= 1
 0 <= w_k10_RIO_s47 <= 1
 0 <= w_k10_RIO_s48 <= 1
 0 <= w_k10_RIO_s49 <= 1
 0 <= w_k10_RIO_s50 <= 1
 0 <= w_k10_RIO_s51 <= 1
 0 <= w_k10_RIO_s52 <= 1
 0 <= w_k10_RIO_s53 <= 1
 0 <= w_k10_RIO_s54 <= 1
 0 <= w_k10_RIO_s55 <= 1
 0 <= w_k10_RIO_s56 <= 1
 0 <= w_k10_RIO_s57 <= 1
 0 <= w_k10_RIO_s58 <= 1
 0 <= w_k10_RIO_s59 <= 1
 0 <= w_k10_RIO_s60 <= 1
 0 <= w_k10_RIO_s61 <= 1
 0 <= w_k10_RIO_s62 <= 1
 0 <= w_k10_RIO_s63 <= 1
 0 <= w_k10_RIO_s64 <= 1
 0 <= w_k10_RIO_s65 <= 1
 0 <= w_k10_RIO_s66 <= 1
 0 <= w_k10_RIO_s67 <= 1
 0 <= w_k10_RIO_s68 <= 1
 0 <= w_k10_RIO_s69 <= 1
 0 <= w_k10_RIO_s70 <= 1
 0 <= w_k10_RIO_s71 <= 1
 0 <= w_k10_RIO_s72 <= 1
 0 <= w_k10_RIO_s73 <= 1
 0 <= w_k10_RIO_s74 <= 1
 0 <= w_k10_RIO_s75 <= 1
 0 <= w_k10_RIO_s76 <= 1
 0 <= w_k10_RIO_s77 <= 1
 0 <= w_k10_RIO_s78 <= 1
 0 <= w_k10_RIO_s79 <= 1
 0 <= w_k10_RIO_s80 <= 1
 0 <= w_k10_RIO_s81 <= 1
 0 <= w_k10_RIO_s82 <= 1
 0 <= w_k10_RIO_s83 <= 1
 0 <= w_k10_RIO_s84 <= 1
 0 <= w_k10_RIO_s85 <= 1
 0 <= w_k10_RIO_s86 <= 1
 0 <= w_k10_RIO_s87 <= 1
 0 <= w_k10_RIO_s88 <= 1
 0 <= w_k10_RIO_s89 <= 1
 0 <= w_k10_RIO_s90 <= 1
 0 <= w_k10_RIO_s91 <= 1
 0 <= w_k10_RIO_s92 <= 1
 0 <= w_k10_RIO_s93 <= 1
 0 <= w_k10_RIO_s94 <= 1
 0 <= w_k10_RIO_s95 <= 1
 0 <= w_k10_RIO_s96 <= 1
 0 <= w_k10_RIO_s97 <= 1
 0 <= w_k10_RIO_s98 <= 1
 0 <= w_k10_RIO_s99 <= 1
 0 <= w_k10_BG_s0 <= 1
 0 <= w_k10_BG_s1 <= 1
 0 <= w_k10_BG_s2 <= 1
 0 <= w_k10_BG_s3 <= 1
 0 <= w_k10_BG_s4 <= 1
 0 <= w_k10_BG_s5 <= 1
 0 <= w_k10_BG_s6 <= 1
 0 <= w_k10_BG_s7 <= 1
 0 <= w_k10_BG_s8 <= 1
 0 <= w_k10_BG_s9 <= 1
 0 <= w_k10_BG_s00 <= 1
 0 <= w_k10_BG_s01 <= 1
 0 <= w_k10_BG_s02 <= 1
 0 <= w_k10_BG_s03 <= 1
 0 <= w_k10_BG_s04 <= 1
 0 <= w_k10_BG_s05 <= 1
 0 <= w_k10_BG_s06 <= 1
 0 <= w_k10_BG_s07 <= 1
 0 <= w_k10_BG_s08 <= 1
 0 <= w_k10_BG_s09 <= 1
 0 <= w_k10_BG_s10 <= 1
 0 <= w_k10_BG_s11 <= 1
 0 <= w_k10_BG_s12 <= 1
 0 <= w_k10_BG_s13 <= 1
 0 <= w_k10_BG_s14 <= 1
 0 <= w_k10_BG_s15 <= 1
 0 <= w_k10_BG_s16 <= 1
 0 <= w_k10_BG_s17 <= 1
 0 <= w_k10_BG_s18 <= 1
 0 <= w_k10_BG_s19 <= 1
 0 <= w_k10_BG_s20 <= 1
 0 <= w_k10_BG_s21 <= 1
 0 <= w_k10_BG_s22 <= 1
 0 <= w_k10_BG_s23 <= 1
 0 <= w_k10_BG_s24 <= 1
 0 <= w_k10_BG_s25 <= 1
 0 <= w_k10_BG_s26 <= 1
 0 <= w_k10_BG_s27 <= 1
 0 <= w_k10_BG_s28 <= 1
 0 <= w_k10_BG_s29 <= 1
 0 <= w_k10_BG_s30 <= 1
 0 <= w_k10_BG_s31 <= 1
 0 <= w_k10_BG_s32 <= 1
 0 <= w_k10_BG_s33 <= 1
 0 <= w_k10_BG_s34 <= 1
 0 <= w_k10_BG_s35 <= 1
 0 <= w_k10_BG_s36 <= 1
 0 <= w_k10_BG_s37 <= 1
 0 <= w_k10_BG_s38 <= 1
 0 <= w_k10_BG_s39 <= 1
 0 <= w_k10_BG_s40 <= 1
 0 <= w_k10_BG_s41 <= 1
 0 <= w_k10_BG_s42 <= 1
 0 <= w_k10_BG_s43 <= 1
 0 <= w_k10_BG_s44 <= 1
 0 <= w_k10_BG_s45 <= 1
 0 <= w_k10_BG_s46 <= 1
 0 <= w_k10_BG_s47 <= 1
 0 <= w_k10_BG_s48 <= 1
 0 <= w_k10_BG_s49 <= 1
 0 <= w_k10_BG_s50 <= 1
 0 <= w_k10_BG_s51 <= 1
 0 <= w_k10_BG_s52 <= 1
 0 <= w_k10_BG_s53 <= 1
 0 <= w_k10_BG_s54 <= 1
 0 <= w_k10_BG_s55 <= 1
 0 <= w_k10_BG_s56 <= 1
 0 <= w_k10_BG_s57 <= 1
 0 <= w_k10_BG_s58 <= 1
 0 <= w_k10_BG_s59 <= 1
 0 <= w_k10_BG_s60 <= 1
 0 <= w_k10_BG_s61 <= 1
 0 <= w_k10_BG_s62 <= 1
 0 <= w_k10_BG_s63 <= 1
 0 <= w_k10_BG_s64 <= 1
 0 <= w_k10_BG_s65 <= 1
 0 <= w_k10_BG_s66 <= 1
 0 <= w_k10_BG_s67 <= 1
 0 <= w_k10_BG_s68 <= 1
 0 <= w_k10_BG_s69 <= 1
 0 <= w_k10_BG_s70 <= 1
 0 <= w_k10_BG_s71 <= 1
 0 <= w_k10_BG_s72 <= 1
 0 <= w_k10_BG_s73 <= 1
 0 <= w_k10_BG_s74 <= 1
 0 <= w_k10_BG_s75 <= 1
 0 <= w_k10_BG_s76 <= 1
 0 <= w_k10_BG_s77 <= 1
 0 <= w_k10_BG_s78 <= 1
 0 <= w_k10_BG_s79 <= 1
 0 <= w_k10_BG_s80 <= 1
 0 <= w_k10_BG_s81 <= 1
 0 <= w_k10_BG_s82 <= 1
 0 <= w_k10_BG_s83 <= 1
 0 <= w_k10_BG_s84 <= 1
 0 <= w_k10_BG_s85 <= 1
 0 <= w_k10_BG_s86 <= 1
 0 <= w_k10_BG_s87 <= 1
 0 <= w_k10_BG_s88 <= 1
 0 <= w_k10_BG_s89 <= 1
 0 <= w_k10_BG_s90 <= 1
 0 <= w_k10_BG_s91 <= 1
 0 <= w_k10_BG_s92 <= 1
 0 <= w_k10_BG_s93 <= 1
 0 <= w_k10_BG_s94 <= 1
 0 <= w_k10_BG_s95 <= 1
 0 <= w_k10_BG_s96 <= 1
 0 <= w_k10_BG_s97 <= 1
 0 <= w_k10_BG_s98 <= 1
 0 <= w_k10_BG_s99 <= 1
 0 <= w_k10_TSCO_s0 <= 1
 0 <= w_k10_TSCO_s1 <= 1
 0 <= w_k10_TSCO_s2 <= 1
 0 <= w_k10_TSCO_s3 <= 1
 0 <= w_k10_TSCO_s4 <= 1
 0 <= w_k10_TSCO_s5 <= 1
 0 <= w_k10_TSCO_s6 <= 1
 0 <= w_k10_TSCO_s7 <= 1
 0 <= w_k10_TSCO_s8 <= 1
 0 <= w_k10_TSCO_s9 <= 1
 0 <= w_k10_TSCO_s00 <= 1
 0 <= w_k10_TSCO_s01 <= 1
 0 <= w_k10_TSCO_s02 <= 1
 0 <= w_k10_TSCO_s03 <= 1
 0 <= w_k10_TSCO_s04 <= 1
 0 <= w_k10_TSCO_s05 <= 1
 0 <= w_k10_TSCO_s06 <= 1
 0 <= w_k10_TSCO_s07 <= 1
 0 <= w_k10_TSCO_s08 <= 1
 0 <= w_k10_TSCO_s09 <= 1
 0 <= w_k10_TSCO_s10 <= 1
 0 <= w_k10_TSCO_s11 <= 1
 0 <= w_k10_TSCO_s12 <= 1
 0 <= w_k10_TSCO_s13 <= 1
 0 <= w_k10_TSCO_s14 <= 1
 0 <= w_k10_TSCO_s15 <= 1
 0 <= w_k10_TSCO_s16 <= 1
 0 <= w_k10_TSCO_s17 <= 1
 0 <= w_k10_TSCO_s18 <= 1
 0 <= w_k10_TSCO_s19 <= 1
 0 <= w_k10_TSCO_s20 <= 1
 0 <= w_k10_TSCO_s21 <= 1
 0 <= w_k10_TSCO_s22 <= 1
 0 <= w_k10_TSCO_s23 <= 1
 0 <= w_k10_TSCO_s24 <= 1
 0 <= w_k10_TSCO_s25 <= 1
 0 <= w_k10_TSCO_s26 <= 1
 0 <= w_k10_TSCO_s27 <= 1
 0 <= w_k10_TSCO_s28 <= 1
 0 <= w_k10_TSCO_s29 <= 1
 0 <= w_k10_TSCO_s30 <= 1
 0 <= w_k10_TSCO_s31 <= 1
 0 <= w_k10_TSCO_s32 <= 1
 0 <= w_k10_TSCO_s33 <= 1
 0 <= w_k10_TSCO_s34 <= 1
 0 <= w_k10_TSCO_s35 <= 1
 0 <= w_k10_TSCO_s36 <= 1
 0 <= w_k10_TSCO_s37 <= 1
 0 <= w_k10_TSCO_s38 <= 1
 0 <= w_k10_TSCO_s39 <= 1
 0 <= w_k10_TSCO_s40 <= 1
 0 <= w_k10_TSCO_s41 <= 1
 0 <= w_k10_TSCO_s42 <= 1
 0 <= w_k10_TSCO_s43 <= 1
 0 <= w_k10_TSCO_s44 <= 1
 0 <= w_k10_TSCO_s45 <= 1
 0 <= w_k10_TSCO_s46 <= 1
 0 <= w_k10_TSCO_s47 <= 1
 0 <= w_k10_TSCO_s48 <= 1
 0 <= w_k10_TSCO_s49 <= 1
 0 <= w_k10_TSCO_s50 <= 1
 0 <= w_k10_TSCO_s51 <= 1
 0 <= w_k10_TSCO_s52 <= 1
 0 <= w_k10_TSCO_s53 <= 1
 0 <= w_k10_TSCO_s54 <= 1
 0 <= w_k10_TSCO_s55 <= 1
 0 <= w_k10_TSCO_s56 <= 1
 0 <= w_k10_TSCO_s57 <= 1
 0 <= w_k10_TSCO_s58 <= 1
 0 <= w_k10_TSCO_s59 <= 1
 0 <= w_k10_TSCO_s60 <= 1
 0 <= w_k10_TSCO_s61 <= 1
 0 <= w_k10_TSCO_s62 <= 1
 0 <= w_k10_TSCO_s63 <= 1
 0 <= w_k10_TSCO_s64 <= 1
 0 <= w_k10_TSCO_s65 <= 1
 0 <= w_k10_TSCO_s66 <= 1
 0 <= w_k10_TSCO_s67 <= 1
 0 <= w_k10_TSCO_s68 <= 1
 0 <= w_k10_TSCO_s69 <= 1
 0 <= w_k10_TSCO_s70 <= 1
 0 <= w_k10_TSCO_s71 <= 1
 0 <= w_k10_TSCO_s72 <= 1
 0 <= w_k10_TSCO_s73 <= 1
 0 <= w_k10_TSCO_s74 <= 1
 0 <= w_k10_TSCO_s75 <= 1
 0 <= w_k10_TSCO_s76 <= 1
 0 <= w_k10_TSCO_s77 <= 1
 0 <= w_k10_TSCO_s78 <= 1
 0 <= w_k10_TSCO_s79 <= 1
 0 <= w_k10_TSCO_s80 <= 1
 0 <= w_k10_TSCO_s81 <= 1
 0 <= w_k10_TSCO_s82 <= 1
 0 <= w_k10_TSCO_s83 <= 1
 0 <= w_k10_TSCO_s84 <= 1
 0 <= w_k10_TSCO_s85 <= 1
 0 <= w_k10_TSCO_s86 <= 1
 0 <= w_k10_TSCO_s87 <= 1
 0 <= w_k10_TSCO_s88 <= 1
 0 <= w_k10_TSCO_s89 <= 1
 0 <= w_k10_TSCO_s90 <= 1
 0 <= w_k10_TSCO_s91 <= 1
 0 <= w_k10_TSCO_s92 <= 1
 0 <= w_k10_TSCO_s93 <= 1
 0 <= w_k10_TSCO_s94 <= 1
 0 <= w_k10_TSCO_s95 <= 1
 0 <= w_k10_TSCO_s96 <= 1
 0 <= w_k10_TSCO_s97 <= 1
 0 <= w_k10_TSCO_s98 <= 1
 0 <= w_k10_TSCO_s99 <= 1
 0 <= w_k10_AAL_s0 <= 1
 0 <= w_k10_AAL_s1 <= 1
 0 <= w_k10_AAL_s2 <= 1
 0 <= w_k10_AAL_s3 <= 1
 0 <= w_k10_AAL_s4 <= 1
 0 <= w_k10_AAL_s5 <= 1
 0 <= w_k10_AAL_s6 <= 1
 0 <= w_k10_AAL_s7 <= 1
 0 <= w_k10_AAL_s8 <= 1
 0 <= w_k10_AAL_s9 <= 1
 0 <= w_k10_AAL_s00 <= 1
 0 <= w_k10_AAL_s01 <= 1
 0 <= w_k10_AAL_s02 <= 1
 0 <= w_k10_AAL_s03 <= 1
 0 <= w_k10_AAL_s04 <= 1
 0 <= w_k10_AAL_s05 <= 1
 0 <= w_k10_AAL_s06 <= 1
 0 <= w_k10_AAL_s07 <= 1
 0 <= w_k10_AAL_s08 <= 1
 0 <= w_k10_AAL_s09 <= 1
 0 <= w_k10_AAL_s10 <= 1
 0 <= w_k10_AAL_s11 <= 1
 0 <= w_k10_AAL_s12 <= 1
 0 <= w_k10_AAL_s13 <= 1
 0 <= w_k10_AAL_s14 <= 1
 0 <= w_k10_AAL_s15 <= 1
 0 <= w_k10_AAL_s16 <= 1
 0 <= w_k10_AAL_s17 <= 1
 0 <= w_k10_AAL_s18 <= 1
 0 <= w_k10_AAL_s19 <= 1
 0 <= w_k10_AAL_s20 <= 1
 0 <= w_k10_AAL_s21 <= 1
 0 <= w_k10_AAL_s22 <= 1
 0 <= w_k10_AAL_s23 <= 1
 0 <= w_k10_AAL_s24 <= 1
 0 <= w_k10_AAL_s25 <= 1
 0 <= w_k10_AAL_s26 <= 1
 0 <= w_k10_AAL_s27 <= 1
 0 <= w_k10_AAL_s28 <= 1
 0 <= w_k10_AAL_s29 <= 1
 0 <= w_k10_AAL_s30 <= 1
 0 <= w_k10_AAL_s31 <= 1
 0 <= w_k10_AAL_s32 <= 1
 0 <= w_k10_AAL_s33 <= 1
 0 <= w_k10_AAL_s34 <= 1
 0 <= w_k10_AAL_s35 <= 1
 0 <= w_k10_AAL_s36 <= 1
 0 <= w_k10_AAL_s37 <= 1
 0 <= w_k10_AAL_s38 <= 1
 0 <= w_k10_AAL_s39 <= 1
 0 <= w_k10_AAL_s40 <= 1
 0 <= w_k10_AAL_s41 <= 1
 0 <= w_k10_AAL_s42 <= 1
 0 <= w_k10_AAL_s43 <= 1
 0 <= w_k10_AAL_s44 <= 1
 0 <= w_k10_AAL_s45 <= 1
 0 <= w_k10_AAL_s46 <= 1
 0 <= w_k10_AAL_s47 <= 1
 0 <= w_k10_AAL_s48 <= 1
 0 <= w_k10_AAL_s49 <= 1
 0 <= w_k10_AAL_s50 <= 1
 0 <= w_k10_AAL_s51 <= 1
 0 <= w_k10_AAL_s52 <= 1
 0 <= w_k10_AAL_s53 <= 1
 0 <= w_k10_AAL_s54 <= 1
 0 <= w_k10_AAL_s55 <= 1
 0 <= w_k10_AAL_s56 <= 1
 0 <= w_k10_AAL_s57 <= 1
 0 <= w_k10_AAL_s58 <= 1
 0 <= w_k10_AAL_s59 <= 1
 0 <= w_k10_AAL_s60 <= 1
 0 <= w_k10_AAL_s61 <= 1
 0 <= w_k10_AAL_s62 <= 1
 0 <= w_k10_AAL_s63 <= 1
 0 <= w_k10_AAL_s64 <= 1
 0 <= w_k10_AAL_s65 <= 1
 0 <= w_k10_AAL_s66 <= 1
 0 <= w_k10_AAL_s67 <= 1
 0 <= w_k10_AAL_s68 <= 1
 0 <= w_k10_AAL_s69 <= 1
 0 <= w_k10_AAL_s70 <= 1
 0 <= w_k10_AAL_s71 <= 1
 0 <= w_k10_AAL_s72 <= 1
 0 <= w_k10_AAL_s73 <= 1
 0 <= w_k10_AAL_s74 <= 1
 0 <= w_k10_AAL_s75 <= 1
 0 <= w_k10_AAL_s76 <= 1
 0 <= w_k10_AAL_s77 <= 1
 0 <= w_k10_AAL_s78 <= 1
 0 <= w_k10_AAL_s79 <= 1
 0 <= w_k10_AAL_s80 <= 1
 0 <= w_k10_AAL_s81 <= 1
 0 <= w_k10_AAL_s82 <= 1
 0 <= w_k10_AAL_s83 <= 1
 0 <= w_k10_AAL_s84 <= 1
 0 <= w_k10_AAL_s85 <= 1
 0 <= w_k10_AAL_s86 <= 1
 0 <= w_k10_AAL_s87 <= 1
 0 <= w_k10_AAL_s88 <= 1
 0 <= w_k10_AAL_s89 <= 1
 0 <= w_k10_AAL_s90 <= 1
 0 <= w_k10_AAL_s91 <= 1
 0 <= w_k10_AAL_s92 <= 1
 0 <= w_k10_AAL_s93 <= 1
 0 <= w_k10_AAL_s94 <= 1
 0 <= w_k10_AAL_s95 <= 1
 0 <= w_k10_AAL_s96 <= 1
 0 <= w_k10_AAL_s97 <= 1
 0 <= w_k10_AAL_s98 <= 1
 0 <= w_k10_AAL_s99 <= 1
 0 <= w_k10_PRU_s0 <= 1
 0 <= w_k10_PRU_s1 <= 1
 0 <= w_k10_PRU_s2 <= 1
 0 <= w_k10_PRU_s3 <= 1
 0 <= w_k10_PRU_s4 <= 1
 0 <= w_k10_PRU_s5 <= 1
 0 <= w_k10_PRU_s6 <= 1
 0 <= w_k10_PRU_s7 <= 1
 0 <= w_k10_PRU_s8 <= 1
 0 <= w_k10_PRU_s9 <= 1
 0 <= w_k10_PRU_s00 <= 1
 0 <= w_k10_PRU_s01 <= 1
 0 <= w_k10_PRU_s02 <= 1
 0 <= w_k10_PRU_s03 <= 1
 0 <= w_k10_PRU_s04 <= 1
 0 <= w_k10_PRU_s05 <= 1
 0 <= w_k10_PRU_s06 <= 1
 0 <= w_k10_PRU_s07 <= 1
 0 <= w_k10_PRU_s08 <= 1
 0 <= w_k10_PRU_s09 <= 1
 0 <= w_k10_PRU_s10 <= 1
 0 <= w_k10_PRU_s11 <= 1
 0 <= w_k10_PRU_s12 <= 1
 0 <= w_k10_PRU_s13 <= 1
 0 <= w_k10_PRU_s14 <= 1
 0 <= w_k10_PRU_s15 <= 1
 0 <= w_k10_PRU_s16 <= 1
 0 <= w_k10_PRU_s17 <= 1
 0 <= w_k10_PRU_s18 <= 1
 0 <= w_k10_PRU_s19 <= 1
 0 <= w_k10_PRU_s20 <= 1
 0 <= w_k10_PRU_s21 <= 1
 0 <= w_k10_PRU_s22 <= 1
 0 <= w_k10_PRU_s23 <= 1
 0 <= w_k10_PRU_s24 <= 1
 0 <= w_k10_PRU_s25 <= 1
 0 <= w_k10_PRU_s26 <= 1
 0 <= w_k10_PRU_s27 <= 1
 0 <= w_k10_PRU_s28 <= 1
 0 <= w_k10_PRU_s29 <= 1
 0 <= w_k10_PRU_s30 <= 1
 0 <= w_k10_PRU_s31 <= 1
 0 <= w_k10_PRU_s32 <= 1
 0 <= w_k10_PRU_s33 <= 1
 0 <= w_k10_PRU_s34 <= 1
 0 <= w_k10_PRU_s35 <= 1
 0 <= w_k10_PRU_s36 <= 1
 0 <= w_k10_PRU_s37 <= 1
 0 <= w_k10_PRU_s38 <= 1
 0 <= w_k10_PRU_s39 <= 1
 0 <= w_k10_PRU_s40 <= 1
 0 <= w_k10_PRU_s41 <= 1
 0 <= w_k10_PRU_s42 <= 1
 0 <= w_k10_PRU_s43 <= 1
 0 <= w_k10_PRU_s44 <= 1
 0 <= w_k10_PRU_s45 <= 1
 0 <= w_k10_PRU_s46 <= 1
 0 <= w_k10_PRU_s47 <= 1
 0 <= w_k10_PRU_s48 <= 1
 0 <= w_k10_PRU_s49 <= 1
 0 <= w_k10_PRU_s50 <= 1
 0 <= w_k10_PRU_s51 <= 1
 0 <= w_k10_PRU_s52 <= 1
 0 <= w_k10_PRU_s53 <= 1
 0 <= w_k10_PRU_s54 <= 1
 0 <= w_k10_PRU_s55 <= 1
 0 <= w_k10_PRU_s56 <= 1
 0 <= w_k10_PRU_s57 <= 1
 0 <= w_k10_PRU_s58 <= 1
 0 <= w_k10_PRU_s59 <= 1
 0 <= w_k10_PRU_s60 <= 1
 0 <= w_k10_PRU_s61 <= 1
 0 <= w_k10_PRU_s62 <= 1
 0 <= w_k10_PRU_s63 <= 1
 0 <= w_k10_PRU_s64 <= 1
 0 <= w_k10_PRU_s65 <= 1
 0 <= w_k10_PRU_s66 <= 1
 0 <= w_k10_PRU_s67 <= 1
 0 <= w_k10_PRU_s68 <= 1
 0 <= w_k10_PRU_s69 <= 1
 0 <= w_k10_PRU_s70 <= 1
 0 <= w_k10_PRU_s71 <= 1
 0 <= w_k10_PRU_s72 <= 1
 0 <= w_k10_PRU_s73 <= 1
 0 <= w_k10_PRU_s74 <= 1
 0 <= w_k10_PRU_s75 <= 1
 0 <= w_k10_PRU_s76 <= 1
 0 <= w_k10_PRU_s77 <= 1
 0 <= w_k10_PRU_s78 <= 1
 0 <= w_k10_PRU_s79 <= 1
 0 <= w_k10_PRU_s80 <= 1
 0 <= w_k10_PRU_s81 <= 1
 0 <= w_k10_PRU_s82 <= 1
 0 <= w_k10_PRU_s83 <= 1
 0 <= w_k10_PRU_s84 <= 1
 0 <= w_k10_PRU_s85 <= 1
 0 <= w_k10_PRU_s86 <= 1
 0 <= w_k10_PRU_s87 <= 1
 0 <= w_k10_PRU_s88 <= 1
 0 <= w_k10_PRU_s89 <= 1
 0 <= w_k10_PRU_s90 <= 1
 0 <= w_k10_PRU_s91 <= 1
 0 <= w_k10_PRU_s92 <= 1
 0 <= w_k10_PRU_s93 <= 1
 0 <= w_k10_PRU_s94 <= 1
 0 <= w_k10_PRU_s95 <= 1
 0 <= w_k10_PRU_s96 <= 1
 0 <= w_k10_PRU_s97 <= 1
 0 <= w_k10_PRU_s98 <= 1
 0 <= w_k10_PRU_s99 <= 1
 0 <= b_0_HSBC <= 0.2
 0 <= b_0_VOD <= 0.2
 0 <= b_0_BP <= 0.2
 0 <= b_0_GSK <= 0.2
 0 <= b_0_AZN <= 0.2
 0 <= b_0_RIO <= 0.2
 0 <= b_0_BG <= 0.2
 0 <= b_0_TSCO <= 0.2
 0 <= b_0_AAL <= 0.2
 0 <= b_0_PRU <= 0.2
 0 <= b_k10_HSBC_s0 <= 0.2
 0 <= b_k10_HSBC_s1 <= 0.2
 0 <= b_k10_HSBC_s2 <= 0.2
 0 <= b_k10_HSBC_s3 <= 0.2
 0 <= b_k10_HSBC_s4 <= 0.2
 0 <= b_k10_HSBC_s5 <= 0.2
 0 <= b_k10_HSBC_s6 <= 0.2
 0 <= b_k10_HSBC_s7 <= 0.2
 0 <= b_k10_HSBC_s8 <= 0.2
 0 <= b_k10_HSBC_s9 <= 0.2
 0 <= b_k10_HSBC_s00 <= 0.2
 0 <= b_k10_HSBC_s01 <= 0.2
 0 <= b_k10_HSBC_s02 <= 0.2
 0 <= b_k10_HSBC_s03 <= 0.2
 0 <= b_k10_HSBC_s04 <= 0.2
 0 <= b_k10_HSBC_s05 <= 0.2
 0 <= b_k10_HSBC_s06 <= 0.2
 0 <= b_k10_HSBC_s07 <= 0.2
 0 <= b_k10_HSBC_s08 <= 0.2
 0 <= b_k10_HSBC_s09 <= 0.2
 0 <= b_k10_HSBC_s10 <= 0.2
 0 <= b_k10_HSBC_s11 <= 0.2
 0 <= b_k10_HSBC_s12 <= 0.2
 0 <= b_k10_HSBC_s13 <= 0.2
 0 <= b_k10_HSBC_s14 <= 0.2
 0 <= b_k10_HSBC_s15 <= 0.2
 0 <= b_k10_HSBC_s16 <= 0.2
 0 <= b_k10_HSBC_s17 <= 0.2
 0 <= b_k10_HSBC_s18 <= 0.2
 0 <= b_k10_HSBC_s19 <= 0.2
 0 <= b_k10_HSBC_s20 <= 0.2
 0 <= b_k10_HSBC_s21 <= 0.2
 0 <= b_k10_HSBC_s22 <= 0.2
 0 <= b_k10_HSBC_s23 <= 0.2
 0 <= b_k10_HSBC_s24 <= 0.2
 0 <= b_k10_HSBC_s25 <= 0.2
 0 <= b_k10_HSBC_s26 <= 0.2
 0 <= b_k10_HSBC_s27 <= 0.2
 0 <= b_k10_HSBC_s28 <= 0.2
 0 <= b_k10_HSBC_s29 <= 0.2
 0 <= b_k10_HSBC_s30 <= 0.2
 0 <= b_k10_HSBC_s31 <= 0.2
 0 <= b_k10_HSBC_s32 <= 0.2
 0 <= b_k10_HSBC_s33 <= 0.2
 0 <= b_k10_HSBC_s34 <= 0.2
 0 <= b_k10_HSBC_s35 <= 0.2
 0 <= b_k10_HSBC_s36 <= 0.2
 0 <= b_k10_HSBC_s37 <= 0.2
 0 <= b_k10_HSBC_s38 <= 0.2
 0 <= b_k10_HSBC_s39 <= 0.2
 0 <= b_k10_HSBC_s40 <= 0.2
 0 <= b_k10_HSBC_s41 <= 0.2
 0 <= b_k10_HSBC_s42 <= 0.2
 0 <= b_k10_HSBC_s43 <= 0.2
 0 <= b_k10_HSBC_s44 <= 0.2
 0 <= b_k10_HSBC_s45 <= 0.2
 0 <= b_k10_HSBC_s46 <= 0.2
 0 <= b_k10_HSBC_s47 <= 0.2
 0 <= b_k10_HSBC_s48 <= 0.2
 0 <= b_k10_HSBC_s49 <= 0.2
 0 <= b_k10_HSBC_s50 <= 0.2
 0 <= b_k10_HSBC_s51 <= 0.2
 0 <= b_k10_HSBC_s52 <= 0.2
 0 <= b_k10_HSBC_s53 <= 0.2
 0 <= b_k10_HSBC_s54 <= 0.2
 0 <= b_k10_HSBC_s55 <= 0.2
 0 <= b_k10_HSBC_s56 <= 0.2
 0 <= b_k10_HSBC_s57 <= 0.2
 0 <= b_k10_HSBC_s58 <= 0.2
 0 <= b_k10_HSBC_s59 <= 0.2
 0 <= b_k10_HSBC_s60 <= 0.2
 0 <= b_k10_HSBC_s61 <= 0.2
 0 <= b_k10_HSBC_s62 <= 0.2
 0 <= b_k10_HSBC_s63 <= 0.2
 0 <= b_k10_HSBC_s64 <= 0.2
 0 <= b_k10_HSBC_s65 <= 0.2
 0 <= b_k10_HSBC_s66 <= 0.2
 0 <= b_k10_HSBC_s67 <= 0.2
 0 <= b_k10_HSBC_s68 <= 0.2
 0 <= b_k10_HSBC_s69 <= 0.2
 0 <= b_k10_HSBC_s70 <= 0.2
 0 <= b_k10_HSBC_s71 <= 0.2
 0 <= b_k10_HSBC_s72 <= 0.2
 0 <= b_k10_HSBC_s73 <= 0.2
 0 <= b_k10_HSBC_s74 <= 0.2
 0 <= b_k10_HSBC_s75 <= 0.2
 0 <= b_k10_HSBC_s76 <= 0.2
 0 <= b_k10_HSBC_s77 <= 0.2
 0 <= b_k10_HSBC_s78 <= 0.2
 0 <= b_k10_HSBC_s79 <= 0.2
 0 <= b_k10_HSBC_s80 <= 0.2
 0 <= b_k10_HSBC_s81 <= 0.2
 0 <= b_k10_HSBC_s82 <= 0.2
 0 <= b_k10_HSBC_s83 <= 0.2
 0 <= b_k10_HSBC_s84 <= 0.2
 0 <= b_k10_HSBC_s85 <= 0.2
 0 <= b_k10_HSBC_s86 <= 0.2
 0 <= b_k10_HSBC_s87 <= 0.2
 0 <= b_k10_HSBC_s88 <= 0.2
 0 <= b_k10_HSBC_s89 <= 0.2
 0 <= b_k10_HSBC_s90 <= 0.2
 0 <= b_k10_HSBC_s91 <= 0.2
 0 <= b_k10_HSBC_s92 <= 0.2
 0 <= b_k10_HSBC_s93 <= 0.2
 0 <= b_k10_HSBC_s94 <= 0.2
 0 <= b_k10_HSBC_s95 <= 0.2
 0 <= b_k10_HSBC_s96 <= 0.2
 0 <= b_k10_HSBC_s97 <= 0.2
 0 <= b_k10_HSBC_s98 <= 0.2
 0 <= b_k10_HSBC_s99 <= 0.2
 0 <= b_k10_VOD_s0 <= 0.2
 0 <= b_k10_VOD_s1 <= 0.2
 0 <= b_k10_VOD_s2 <= 0.2
 0 <= b_k10_VOD_s3 <= 0.2
 0 <= b_k10_VOD_s4 <= 0.2
 0 <= b_k10_VOD_s5 <= 0.2
 0 <= b_k10_VOD_s6 <= 0.2
 0 <= b_k10_VOD_s7 <= 0.2
 0 <= b_k10_VOD_s8 <= 0.2
 0 <= b_k10_VOD_s9 <= 0.2
 0 <= b_k10_VOD_s00 <= 0.2
 0 <= b_k10_VOD_s01 <= 0.2
 0 <= b_k10_VOD_s02 <= 0.2
 0 <= b_k10_VOD_s03 <= 0.2
 0 <= b_k10_VOD_s04 <= 0.2
 0 <= b_k10_VOD_s05 <= 0.2
 0 <= b_k10_VOD_s06 <= 0.2
 0 <= b_k10_VOD_s07 <= 0.2
 0 <= b_k10_VOD_s08 <= 0.2
 0 <= b_k10_VOD_s09 <= 0.2
 0 <= b_k10_VOD_s10 <= 0.2
 0 <= b_k10_VOD_s11 <= 0.2
 0 <= b_k10_VOD_s12 <= 0.2
 0 <= b_k10_VOD_s13 <= 0.2
 0 <= b_k10_VOD_s14 <= 0.2
 0 <= b_k10_VOD_s15 <= 0.2
 0 <= b_k10_VOD_s16 <= 0.2
 0 <= b_k10_VOD_s17 <= 0.2
 0 <= b_k10_VOD_s18 <= 0.2
 0 <= b_k10_VOD_s19 <= 0.2
 0 <= b_k10_VOD_s20 <= 0.2
 0 <= b_k10_VOD_s21 <= 0.2
 0 <= b_k10_VOD_s22 <= 0.2
 0 <= b_k10_VOD_s23 <= 0.2
 0 <= b_k10_VOD_s24 <= 0.2
 0 <= b_k10_VOD_s25 <= 0.2
 0 <= b_k10_VOD_s26 <= 0.2
 0 <= b_k10_VOD_s27 <= 0.2
 0 <= b_k10_VOD_s28 <= 0.2
 0 <= b_k10_VOD_s29 <= 0.2
 0 <= b_k10_VOD_s30 <= 0.2
 0 <= b_k10_VOD_s31 <= 0.2
 0 <= b_k10_VOD_s32 <= 0.2
 0 <= b_k10_VOD_s33 <= 0.2
 0 <= b_k10_VOD_s34 <= 0.2
 0 <= b_k10_VOD_s35 <= 0.2
 0 <= b_k10_VOD_s36 <= 0.2
 0 <= b_k10_VOD_s37 <= 0.2
 0 <= b_k10_VOD_s38 <= 0.2
 0 <= b_k10_VOD_s39 <= 0.2
 0 <= b_k10_VOD_s40 <= 0.2
 0 <= b_k10_VOD_s41 <= 0.2
 0 <= b_k10_VOD_s42 <= 0.2
 0 <= b_k10_VOD_s43 <= 0.2
 0 <= b_k10_VOD_s44 <= 0.2
 0 <= b_k10_VOD_s45 <= 0.2
 0 <= b_k10_VOD_s46 <= 0.2
 0 <= b_k10_VOD_s47 <= 0.2
 0 <= b_k10_VOD_s48 <= 0.2
 0 <= b_k10_VOD_s49 <= 0.2
 0 <= b_k10_VOD_s50 <= 0.2
 0 <= b_k10_VOD_s51 <= 0.2
 0 <= b_k10_VOD_s52 <= 0.2
 0 <= b_k10_VOD_s53 <= 0.2
 0 <= b_k10_VOD_s54 <= 0.2
 0 <= b_k10_VOD_s55 <= 0.2
 0 <= b_k10_VOD_s56 <= 0.2
 0 <= b_k10_VOD_s57 <= 0.2
 0 <= b_k10_VOD_s58 <= 0.2
 0 <= b_k10_VOD_s59 <= 0.2
 0 <= b_k10_VOD_s60 <= 0.2
 0 <= b_k10_VOD_s61 <= 0.2
 0 <= b_k10_VOD_s62 <= 0.2
 0 <= b_k10_VOD_s63 <= 0.2
 0 <= b_k10_VOD_s64 <= 0.2
 0 <= b_k10_VOD_s65 <= 0.2
 0 <= b_k10_VOD_s66 <= 0.2
 0 <= b_k10_VOD_s67 <= 0.2
 0 <= b_k10_VOD_s68 <= 0.2
 0 <= b_k10_VOD_s69 <= 0.2
 0 <= b_k10_VOD_s70 <= 0.2
 0 <= b_k10_VOD_s71 <= 0.2
 0 <= b_k10_VOD_s72 <= 0.2
 0 <= b_k10_VOD_s73 <= 0.2
 0 <= b_k10_VOD_s74 <= 0.2
 0 <= b_k10_VOD_s75 <= 0.2
 0 <= b_k10_VOD_s76 <= 0.2
 0 <= b_k10_VOD_s77 <= 0.2
 0 <= b_k10_VOD_s78 <= 0.2
 0 <= b_k10_VOD_s79 <= 0.2
 0 <= b_k10_VOD_s80 <= 0.2
 0 <= b_k10_VOD_s81 <= 0.2
 0 <= b_k10_VOD_s82 <= 0.2
 0 <= b_k10_VOD_s83 <= 0.2
 0 <= b_k10_VOD_s84 <= 0.2
 0 <= b_k10_VOD_s85 <= 0.2
 0 <= b_k10_VOD_s86 <= 0.2
 0 <= b_k10_VOD_s87 <= 0.2
 0 <= b_k10_VOD_s88 <= 0.2
 0 <= b_k10_VOD_s89 <= 0.2
 0 <= b_k10_VOD_s90 <= 0.2
 0 <= b_k10_VOD_s91 <= 0.2
 0 <= b_k10_VOD_s92 <= 0.2
 0 <= b_k10_VOD_s93 <= 0.2
 0 <= b_k10_VOD_s94 <= 0.2
 0 <= b_k10_VOD_s95 <= 0.2
 0 <= b_k10_VOD_s96 <= 0.2
 0 <= b_k10_VOD_s97 <= 0.2
 0 <= b_k10_VOD_s98 <= 0.2
 0 <= b_k10_VOD_s99 <= 0.2
 0 <= b_k10_BP_s0 <= 0.2
 0 <= b_k10_BP_s1 <= 0.2
 0 <= b_k10_BP_s2 <= 0.2
 0 <= b_k10_BP_s3 <= 0.2
 0 <= b_k10_BP_s4 <= 0.2
 0 <= b_k10_BP_s5 <= 0.2
 0 <= b_k10_BP_s6 <= 0.2
 0 <= b_k10_BP_s7 <= 0.2
 0 <= b_k10_BP_s8 <= 0.2
 0 <= b_k10_BP_s9 <= 0.2
 0 <= b_k10_BP_s00 <= 0.2
 0 <= b_k10_BP_s01 <= 0.2
 0 <= b_k10_BP_s02 <= 0.2
 0 <= b_k10_BP_s03 <= 0.2
 0 <= b_k10_BP_s04 <= 0.2
 0 <= b_k10_BP_s05 <= 0.2
 0 <= b_k10_BP_s06 <= 0.2
 0 <= b_k10_BP_s07 <= 0.2
 0 <= b_k10_BP_s08 <= 0.2
 0 <= b_k10_BP_s09 <= 0.2
 0 <= b_k10_BP_s10 <= 0.2
 0 <= b_k10_BP_s11 <= 0.2
 0 <= b_k10_BP_s12 <= 0.2
 0 <= b_k10_BP_s13 <= 0.2
 0 <= b_k10_BP_s14 <= 0.2
 0 <= b_k10_BP_s15 <= 0.2
 0 <= b_k10_BP_s16 <= 0.2
 0 <= b_k10_BP_s17 <= 0.2
 0 <= b_k10_BP_s18 <= 0.2
 0 <= b_k10_BP_s19 <= 0.2
 0 <= b_k10_BP_s20 <= 0.2
 0 <= b_k10_BP_s21 <= 0.2
 0 <= b_k10_BP_s22 <= 0.2
 0 <= b_k10_BP_s23 <= 0.2
 0 <= b_k10_BP_s24 <= 0.2
 0 <= b_k10_BP_s25 <= 0.2
 0 <= b_k10_BP_s26 <= 0.2
 0 <= b_k10_BP_s27 <= 0.2
 0 <= b_k10_BP_s28 <= 0.2
 0 <= b_k10_BP_s29 <= 0.2
 0 <= b_k10_BP_s30 <= 0.2
 0 <= b_k10_BP_s31 <= 0.2
 0 <= b_k10_BP_s32 <= 0.2
 0 <= b_k10_BP_s33 <= 0.2
 0 <= b_k10_BP_s34 <= 0.2
 0 <= b_k10_BP_s35 <= 0.2
 0 <= b_k10_BP_s36 <= 0.2
 0 <= b_k10_BP_s37 <= 0.2
 0 <= b_k10_BP_s38 <= 0.2
 0 <= b_k10_BP_s39 <= 0.2
 0 <= b_k10_BP_s40 <= 0.2
 0 <= b_k10_BP_s41 <= 0.2
 0 <= b_k10_BP_s42 <= 0.2
 0 <= b_k10_BP_s43 <= 0.2
 0 <= b_k10_BP_s44 <= 0.2
 0 <= b_k10_BP_s45 <= 0.2
 0 <= b_k10_BP_s46 <= 0.2
 0 <= b_k10_BP_s47 <= 0.2
 0 <= b_k10_BP_s48 <= 0.2
 0 <= b_k10_BP_s49 <= 0.2
 0 <= b_k10_BP_s50 <= 0.2
 0 <= b_k10_BP_s51 <= 0.2
 0 <= b_k10_BP_s52 <= 0.2
 0 <= b_k10_BP_s53 <= 0.2
 0 <= b_k10_BP_s54 <= 0.2
 0 <= b_k10_BP_s55 <= 0.2
 0 <= b_k10_BP_s56 <= 0.2
 0 <= b_k10_BP_s57 <= 0.2
 0 <= b_k10_BP_s58 <= 0.2
 0 <= b_k10_BP_s59 <= 0.2
 0 <= b_k10_BP_s60 <= 0.2
 0 <= b_k10_BP_s61 <= 0.2
 0 <= b_k10_BP_s62 <= 0.2
 0 <= b_k10_BP_s63 <= 0.2
 0 <= b_k10_BP_s64 <= 0.2
 0 <= b_k10_BP_s65 <= 0.2
 0 <= b_k10_BP_s66 <= 0.2
 0 <= b_k10_BP_s67 <= 0.2
 0 <= b_k10_BP_s68 <= 0.2
 0 <= b_k10_BP_s69 <= 0.2
 0 <= b_k10_BP_s70 <= 0.2
 0 <= b_k10_BP_s71 <= 0.2
 0 <= b_k10_BP_s72 <= 0.2
 0 <= b_k10_BP_s73 <= 0.2
 0 <= b_k10_BP_s74 <= 0.2
 0 <= b_k10_BP_s75 <= 0.2
 0 <= b_k10_BP_s76 <= 0.2
 0 <= b_k10_BP_s77 <= 0.2
 0 <= b_k10_BP_s78 <= 0.2
 0 <= b_k10_BP_s79 <= 0.2
 0 <= b_k10_BP_s80 <= 0.2
 0 <= b_k10_BP_s81 <= 0.2
 0 <= b_k10_BP_s82 <= 0.2
 0 <= b_k10_BP_s83 <= 0.2
 0 <= b_k10_BP_s84 <= 0.2
 0 <= b_k10_BP_s85 <= 0.2
 0 <= b_k10_BP_s86 <= 0.2
 0 <= b_k10_BP_s87 <= 0.2
 0 <= b_k10_BP_s88 <= 0.2
 0 <= b_k10_BP_s89 <= 0.2
 0 <= b_k10_BP_s90 <= 0.2
 0 <= b_k10_BP_s91 <= 0.2
 0 <= b_k10_BP_s92 <= 0.2
 0 <= b_k10_BP_s93 <= 0.2
 0 <= b_k10_BP_s94 <= 0.2
 0 <= b_k10_BP_s95 <= 0.2
 0 <= b_k10_BP_s96 <= 0.2
 0 <= b_k10_BP_s97 <= 0.2
 0 <= b_k10_BP_s98 <= 0.2
 0 <= b_k10_BP_s99 <= 0.2
 0 <= b_k10_GSK_s0 <= 0.2
 0 <= b_k10_GSK_s1 <= 0.2
 0 <= b_k10_GSK_s2 <= 0.2
 0 <= b_k10_GSK_s3 <= 0.2
 0 <= b_k10_GSK_s4 <= 0.2
 0 <= b_k10_GSK_s5 <= 0.2
 0 <= b_k10_GSK_s6 <= 0.2
 0 <= b_k10_GSK_s7 <= 0.2
 0 <= b_k10_GSK_s8 <= 0.2
 0 <= b_k10_GSK_s9 <= 0.2
 0 <= b_k10_GSK_s00 <= 0.2
 0 <= b_k10_GSK_s01 <= 0.2
 0 <= b_k10_GSK_s02 <= 0.2
 0 <= b_k10_GSK_s03 <= 0.2
 0 <= b_k10_GSK_s04 <= 0.2
 0 <= b_k10_GSK_s05 <= 0.2
 0 <= b_k10_GSK_s06 <= 0.2
 0 <= b_k10_GSK_s07 <= 0.2
 0 <= b_k10_GSK_s08 <= 0.2
 0 <= b_k10_GSK_s09 <= 0.2
 0 <= b_k10_GSK_s10 <= 0.2
 0 <= b_k10_GSK_s11 <= 0.2
 0 <= b_k10_GSK_s12 <= 0.2
 0 <= b_k10_GSK_s13 <= 0.2
 0 <= b_k10_GSK_s14 <= 0.2
 0 <= b_k10_GSK_s15 <= 0.2
 0 <= b_k10_GSK_s16 <= 0.2
 0 <= b_k10_GSK_s17 <= 0.2
 0 <= b_k10_GSK_s18 <= 0.2
 0 <= b_k10_GSK_s19 <= 0.2
 0 <= b_k10_GSK_s20 <= 0.2
 0 <= b_k10_GSK_s21 <= 0.2
 0 <= b_k10_GSK_s22 <= 0.2
 0 <= b_k10_GSK_s23 <= 0.2
 0 <= b_k10_GSK_s24 <= 0.2
 0 <= b_k10_GSK_s25 <= 0.2
 0 <= b_k10_GSK_s26 <= 0.2
 0 <= b_k10_GSK_s27 <= 0.2
 0 <= b_k10_GSK_s28 <= 0.2
 0 <= b_k10_GSK_s29 <= 0.2
 0 <= b_k10_GSK_s30 <= 0.2
 0 <= b_k10_GSK_s31 <= 0.2
 0 <= b_k10_GSK_s32 <= 0.2
 0 <= b_k10_GSK_s33 <= 0.2
 0 <= b_k10_GSK_s34 <= 0.2
 0 <= b_k10_GSK_s35 <= 0.2
 0 <= b_k10_GSK_s36 <= 0.2
 0 <= b_k10_GSK_s37 <= 0.2
 0 <= b_k10_GSK_s38 <= 0.2
 0 <= b_k10_GSK_s39 <= 0.2
 0 <= b_k10_GSK_s40 <= 0.2
 0 <= b_k10_GSK_s41 <= 0.2
 0 <= b_k10_GSK_s42 <= 0.2
 0 <= b_k10_GSK_s43 <= 0.2
 0 <= b_k10_GSK_s44 <= 0.2
 0 <= b_k10_GSK_s45 <= 0.2
 0 <= b_k10_GSK_s46 <= 0.2
 0 <= b_k10_GSK_s47 <= 0.2
 0 <= b_k10_GSK_s48 <= 0.2
 0 <= b_k10_GSK_s49 <= 0.2
 0 <= b_k10_GSK_s50 <= 0.2
 0 <= b_k10_GSK_s51 <= 0.2
 0 <= b_k10_GSK_s52 <= 0.2
 0 <= b_k10_GSK_s53 <= 0.2
 0 <= b_k10_GSK_s54 <= 0.2
 0 <= b_k10_GSK_s55 <= 0.2
 0 <= b_k10_GSK_s56 <= 0.2
 0 <= b_k10_GSK_s57 <= 0.2
 0 <= b_k10_GSK_s58 <= 0.2
 0 <= b_k10_GSK_s59 <= 0.2
 0 <= b_k10_GSK_s60 <= 0.2
 0 <= b_k10_GSK_s61 <= 0.2
 0 <= b_k10_GSK_s62 <= 0.2
 0 <= b_k10_GSK_s63 <= 0.2
 0 <= b_k10_GSK_s64 <= 0.2
 0 <= b_k10_GSK_s65 <= 0.2
 0 <= b_k10_GSK_s66 <= 0.2
 0 <= b_k10_GSK_s67 <= 0.2
 0 <= b_k10_GSK_s68 <= 0.2
 0 <= b_k10_GSK_s69 <= 0.2
 0 <= b_k10_GSK_s70 <= 0.2
 0 <= b_k10_GSK_s71 <= 0.2
 0 <= b_k10_GSK_s72 <= 0.2
 0 <= b_k10_GSK_s73 <= 0.2
 0 <= b_k10_GSK_s74 <= 0.2
 0 <= b_k10_GSK_s75 <= 0.2
 0 <= b_k10_GSK_s76 <= 0.2
 0 <= b_k10_GSK_s77 <= 0.2
 0 <= b_k10_GSK_s78 <= 0.2
 0 <= b_k10_GSK_s79 <= 0.2
 0 <= b_k10_GSK_s80 <= 0.2
 0 <= b_k10_GSK_s81 <= 0.2
 0 <= b_k10_GSK_s82 <= 0.2
 0 <= b_k10_GSK_s83 <= 0.2
 0 <= b_k10_GSK_s84 <= 0.2
 0 <= b_k10_GSK_s85 <= 0.2
 0 <= b_k10_GSK_s86 <= 0.2
 0 <= b_k10_GSK_s87 <= 0.2
 0 <= b_k10_GSK_s88 <= 0.2
 0 <= b_k10_GSK_s89 <= 0.2
 0 <= b_k10_GSK_s90 <= 0.2
 0 <= b_k10_GSK_s91 <= 0.2
 0 <= b_k10_GSK_s92 <= 0.2
 0 <= b_k10_GSK_s93 <= 0.2
 0 <= b_k10_GSK_s94 <= 0.2
 0 <= b_k10_GSK_s95 <= 0.2
 0 <= b_k10_GSK_s96 <= 0.2
 0 <= b_k10_GSK_s97 <= 0.2
 0 <= b_k10_GSK_s98 <= 0.2
 0 <= b_k10_GSK_s99 <= 0.2
 0 <= b_k10_AZN_s0 <= 0.2
 0 <= b_k10_AZN_s1 <= 0.2
 0 <= b_k10_AZN_s2 <= 0.2
 0 <= b_k10_AZN_s3 <= 0.2
 0 <= b_k10_AZN_s4 <= 0.2
 0 <= b_k10_AZN_s5 <= 0.2
 0 <= b_k10_AZN_s6 <= 0.2
 0 <= b_k10_AZN_s7 <= 0.2
 0 <= b_k10_AZN_s8 <= 0.2
 0 <= b_k10_AZN_s9 <= 0.2
 0 <= b_k10_AZN_s00 <= 0.2
 0 <= b_k10_AZN_s01 <= 0.2
 0 <= b_k10_AZN_s02 <= 0.2
 0 <= b_k10_AZN_s03 <= 0.2
 0 <= b_k10_AZN_s04 <= 0.2
 0 <= b_k10_AZN_s05 <= 0.2
 0 <= b_k10_AZN_s06 <= 0.2
 0 <= b_k10_AZN_s07 <= 0.2
 0 <= b_k10_AZN_s08 <= 0.2
 0 <= b_k10_AZN_s09 <= 0.2
 0 <= b_k10_AZN_s10 <= 0.2
 0 <= b_k10_AZN_s11 <= 0.2
 0 <= b_k10_AZN_s12 <= 0.2
 0 <= b_k10_AZN_s13 <= 0.2
 0 <= b_k10_AZN_s14 <= 0.2
 0 <= b_k10_AZN_s15 <= 0.2
 0 <= b_k10_AZN_s16 <= 0.2
 0 <= b_k10_AZN_s17 <= 0.2
 0 <= b_k10_AZN_s18 <= 0.2
 0 <= b_k10_AZN_s19 <= 0.2
 0 <= b_k10_AZN_s20 <= 0.2
 0 <= b_k10_AZN_s21 <= 0.2
 0 <= b_k10_AZN_s22 <= 0.2
 0 <= b_k10_AZN_s23 <= 0.2
 0 <= b_k10_AZN_s24 <= 0.2
 0 <= b_k10_AZN_s25 <= 0.2
 0 <= b_k10_AZN_s26 <= 0.2
 0 <= b_k10_AZN_s27 <= 0.2
 0 <= b_k10_AZN_s28 <= 0.2
 0 <= b_k10_AZN_s29 <= 0.2
 0 <= b_k10_AZN_s30 <= 0.2
 0 <= b_k10_AZN_s31 <= 0.2
 0 <= b_k10_AZN_s32 <= 0.2
 0 <= b_k10_AZN_s33 <= 0.2
 0 <= b_k10_AZN_s34 <= 0.2
 0 <= b_k10_AZN_s35 <= 0.2
 0 <= b_k10_AZN_s36 <= 0.2
 0 <= b_k10_AZN_s37 <= 0.2
 0 <= b_k10_AZN_s38 <= 0.2
 0 <= b_k10_AZN_s39 <= 0.2
 0 <= b_k10_AZN_s40 <= 0.2
 0 <= b_k10_AZN_s41 <= 0.2
 0 <= b_k10_AZN_s42 <= 0.2
 0 <= b_k10_AZN_s43 <= 0.2
 0 <= b_k10_AZN_s44 <= 0.2
 0 <= b_k10_AZN_s45 <= 0.2
 0 <= b_k10_AZN_s46 <= 0.2
 0 <= b_k10_AZN_s47 <= 0.2
 0 <= b_k10_AZN_s48 <= 0.2
 0 <= b_k10_AZN_s49 <= 0.2
 0 <= b_k10_AZN_s50 <= 0.2
 0 <= b_k10_AZN_s51 <= 0.2
 0 <= b_k10_AZN_s52 <= 0.2
 0 <= b_k10_AZN_s53 <= 0.2
 0 <= b_k10_AZN_s54 <= 0.2
 0 <= b_k10_AZN_s55 <= 0.2
 0 <= b_k10_AZN_s56 <= 0.2
 0 <= b_k10_AZN_s57 <= 0.2
 0 <= b_k10_AZN_s58 <= 0.2
 0 <= b_k10_AZN_s59 <= 0.2
 0 <= b_k10_AZN_s60 <= 0.2
 0 <= b_k10_AZN_s61 <= 0.2
 0 <= b_k10_AZN_s62 <= 0.2
 0 <= b_k10_AZN_s63 <= 0.2
 0 <= b_k10_AZN_s64 <= 0.2
 0 <= b_k10_AZN_s65 <= 0.2
 0 <= b_k10_AZN_s66 <= 0.2
 0 <= b_k10_AZN_s67 <= 0.2
 0 <= b_k10_AZN_s68 <= 0.2
 0 <= b_k10_AZN_s69 <= 0.2
 0 <= b_k10_AZN_s70 <= 0.2
 0 <= b_k10_AZN_s71 <= 0.2
 0 <= b_k10_AZN_s72 <= 0.2
 0 <= b_k10_AZN_s73 <= 0.2
 0 <= b_k10_AZN_s74 <= 0.2
 0 <= b_k10_AZN_s75 <= 0.2
 0 <= b_k10_AZN_s76 <= 0.2
 0 <= b_k10_AZN_s77 <= 0.2
 0 <= b_k10_AZN_s78 <= 0.2
 0 <= b_k10_AZN_s79 <= 0.2
 0 <= b_k10_AZN_s80 <= 0.2
 0 <= b_k10_AZN_s81 <= 0.2
 0 <= b_k10_AZN_s82 <= 0.2
 0 <= b_k10_AZN_s83 <= 0.2
 0 <= b_k10_AZN_s84 <= 0.2
 0 <= b_k10_AZN_s85 <= 0.2
 0 <= b_k10_AZN_s86 <= 0.2
 0 <= b_k10_AZN_s87 <= 0.2
 0 <= b_k10_AZN_s88 <= 0.2
 0 <= b_k10_AZN_s89 <= 0.2
 0 <= b_k10_AZN_s90 <= 0.2
 0 <= b_k10_AZN_s91 <= 0.2
 0 <= b_k10_AZN_s92 <= 0.2
 0 <= b_k10_AZN_s93 <= 0.2
 0 <= b_k10_AZN_s94 <= 0.2
 0 <= b_k10_AZN_s95 <= 0.2
 0 <= b_k10_AZN_s96 <= 0.2
 0 <= b_k10_AZN_s97 <= 0.2
 0 <= b_k10_AZN_s98 <= 0.2
 0 <= b_k10_AZN_s99 <= 0.2
 0 <= b_k10_RIO_s0 <= 0.2
 0 <= b_k10_RIO_s1 <= 0.2
 0 <= b_k10_RIO_s2 <= 0.2
 0 <= b_k10_RIO_s3 <= 0.2
 0 <= b_k10_RIO_s4 <= 0.2
 0 <= b_k10_RIO_s5 <= 0.2
 0 <= b_k10_RIO_s6 <= 0.2
 0 <= b_k10_RIO_s7 <= 0.2
 0 <= b_k10_RIO_s8 <= 0.2
 0 <= b_k10_RIO_s9 <= 0.2
 0 <= b_k10_RIO_s00 <= 0.2
 0 <= b_k10_RIO_s01 <= 0.2
 0 <= b_k10_RIO_s02 <= 0.2
 0 <= b_k10_RIO_s03 <= 0.2
 0 <= b_k10_RIO_s04 <= 0.2
 0 <= b_k10_RIO_s05 <= 0.2
 0 <= b_k10_RIO_s06 <= 0.2
 0 <= b_k10_RIO_s07 <= 0.2
 0 <= b_k10_RIO_s08 <= 0.2
 0 <= b_k10_RIO_s09 <= 0.2
 0 <= b_k10_RIO_s10 <= 0.2
 0 <= b_k10_RIO_s11 <= 0.2
 0 <= b_k10_RIO_s12 <= 0.2
 0 <= b_k10_RIO_s13 <= 0.2
 0 <= b_k10_RIO_s14 <= 0.2
 0 <= b_k10_RIO_s15 <= 0.2
 0 <= b_k10_RIO_s16 <= 0.2
 0 <= b_k10_RIO_s17 <= 0.2
 0 <= b_k10_RIO_s18 <= 0.2
 0 <= b_k10_RIO_s19 <= 0.2
 0 <= b_k10_RIO_s20 <= 0.2
 0 <= b_k10_RIO_s21 <= 0.2
 0 <= b_k10_RIO_s22 <= 0.2
 0 <= b_k10_RIO_s23 <= 0.2
 0 <= b_k10_RIO_s24 <= 0.2
 0 <= b_k10_RIO_s25 <= 0.2
 0 <= b_k10_RIO_s26 <= 0.2
 0 <= b_k10_RIO_s27 <= 0.2
 0 <= b_k10_RIO_s28 <= 0.2
 0 <= b_k10_RIO_s29 <= 0.2
 0 <= b_k10_RIO_s30 <= 0.2
 0 <= b_k10_RIO_s31 <= 0.2
 0 <= b_k10_RIO_s32 <= 0.2
 0 <= b_k10_RIO_s33 <= 0.2
 0 <= b_k10_RIO_s34 <= 0.2
 0 <= b_k10_RIO_s35 <= 0.2
 0 <= b_k10_RIO_s36 <= 0.2
 0 <= b_k10_RIO_s37 <= 0.2
 0 <= b_k10_RIO_s38 <= 0.2
 0 <= b_k10_RIO_s39 <= 0.2
 0 <= b_k10_RIO_s40 <= 0.2
 0 <= b_k10_RIO_s41 <= 0.2
 0 <= b_k10_RIO_s42 <= 0.2
 0 <= b_k10_RIO_s43 <= 0.2
 0 <= b_k10_RIO_s44 <= 0.2
 0 <= b_k10_RIO_s45 <= 0.2
 0 <= b_k10_RIO_s46 <= 0.2
 0 <= b_k10_RIO_s47 <= 0.2
 0 <= b_k10_RIO_s48 <= 0.2
 0 <= b_k10_RIO_s49 <= 0.2
 0 <= b_k10_RIO_s50 <= 0.2
 0 <= b_k10_RIO_s51 <= 0.2
 0 <= b_k10_RIO_s52 <= 0.2
 0 <= b_k10_RIO_s53 <= 0.2
 0 <= b_k10_RIO_s54 <= 0.2
 0 <= b_k10_RIO_s55 <= 0.2
 0 <= b_k10_RIO_s56 <= 0.2
 0 <= b_k10_RIO_s57 <= 0.2
 0 <= b_k10_RIO_s58 <= 0.2
 0 <= b_k10_RIO_s59 <= 0.2
 0 <= b_k10_RIO_s60 <= 0.2
 0 <= b_k10_RIO_s61 <= 0.2
 0 <= b_k10_RIO_s62 <= 0.2
 0 <= b_k10_RIO_s63 <= 0.2
 0 <= b_k10_RIO_s64 <= 0.2
 0 <= b_k10_RIO_s65 <= 0.2
 0 <= b_k10_RIO_s66 <= 0.2
 0 <= b_k10_RIO_s67 <= 0.2
 0 <= b_k10_RIO_s68 <= 0.2
 0 <= b_k10_RIO_s69 <= 0.2
 0 <= b_k10_RIO_s70 <= 0.2
 0 <= b_k10_RIO_s71 <= 0.2
 0 <= b_k10_RIO_s72 <= 0.2
 0 <= b_k10_RIO_s73 <= 0.2
 0 <= b_k10_RIO_s74 <= 0.2
 0 <= b_k10_RIO_s75 <= 0.2
 0 <= b_k10_RIO_s76 <= 0.2
 0 <= b_k10_RIO_s77 <= 0.2
 0 <= b_k10_RIO_s78 <= 0.2
 0 <= b_k10_RIO_s79 <= 0.2
 0 <= b_k10_RIO_s80 <= 0.2
 0 <= b_k10_RIO_s81 <= 0.2
 0 <= b_k10_RIO_s82 <= 0.2
 0 <= b_k10_RIO_s83 <= 0.2
 0 <= b_k10_RIO_s84 <= 0.2
 0 <= b_k10_RIO_s85 <= 0.2
 0 <= b_k10_RIO_s86 <= 0.2
 0 <= b_k10_RIO_s87 <= 0.2
 0 <= b_k10_RIO_s88 <= 0.2
 0 <= b_k10_RIO_s89 <= 0.2
 0 <= b_k10_RIO_s90 <= 0.2
 0 <= b_k10_RIO_s91 <= 0.2
 0 <= b_k10_RIO_s92 <= 0.2
 0 <= b_k10_RIO_s93 <= 0.2
 0 <= b_k10_RIO_s94 <= 0.2
 0 <= b_k10_RIO_s95 <= 0.2
 0 <= b_k10_RIO_s96 <= 0.2
 0 <= b_k10_RIO_s97 <= 0.2
 0 <= b_k10_RIO_s98 <= 0.2
 0 <= b_k10_RIO_s99 <= 0.2
 0 <= b_k10_BG_s0 <= 0.2
 0 <= b_k10_BG_s1 <= 0.2
 0 <= b_k10_BG_s2 <= 0.2
 0 <= b_k10_BG_s3 <= 0.2
 0 <= b_k10_BG_s4 <= 0.2
 0 <= b_k10_BG_s5 <= 0.2
 0 <= b_k10_BG_s6 <= 0.2
 0 <= b_k10_BG_s7 <= 0.2
 0 <= b_k10_BG_s8 <= 0.2
 0 <= b_k10_BG_s9 <= 0.2
 0 <= b_k10_BG_s00 <= 0.2
 0 <= b_k10_BG_s01 <= 0.2
 0 <= b_k10_BG_s02 <= 0.2
 0 <= b_k10_BG_s03 <= 0.2
 0 <= b_k10_BG_s04 <= 0.2
 0 <= b_k10_BG_s05 <= 0.2
 0 <= b_k10_BG_s06 <= 0.2
 0 <= b_k10_BG_s07 <= 0.2
 0 <= b_k10_BG_s08 <= 0.2
 0 <= b_k10_BG_s09 <= 0.2
 0 <= b_k10_BG_s10 <= 0.2
 0 <= b_k10_BG_s11 <= 0.2
 0 <= b_k10_BG_s12 <= 0.2
 0 <= b_k10_BG_s13 <= 0.2
 0 <= b_k10_BG_s14 <= 0.2
 0 <= b_k10_BG_s15 <= 0.2
 0 <= b_k10_BG_s16 <= 0.2
 0 <= b_k10_BG_s17 <= 0.2
 0 <= b_k10_BG_s18 <= 0.2
 0 <= b_k10_BG_s19 <= 0.2
 0 <= b_k10_BG_s20 <= 0.2
 0 <= b_k10_BG_s21 <= 0.2
 0 <= b_k10_BG_s22 <= 0.2
 0 <= b_k10_BG_s23 <= 0.2
 0 <= b_k10_BG_s24 <= 0.2
 0 <= b_k10_BG_s25 <= 0.2
 0 <= b_k10_BG_s26 <= 0.2
 0 <= b_k10_BG_s27 <= 0.2
 0 <= b_k10_BG_s28 <= 0.2
 0 <= b_k10_BG_s29 <= 0.2
 0 <= b_k10_BG_s30 <= 0.2
 0 <= b_k10_BG_s31 <= 0.2
 0 <= b_k10_BG_s32 <= 0.2
 0 <= b_k10_BG_s33 <= 0.2
 0 <= b_k10_BG_s34 <= 0.2
 0 <= b_k10_BG_s35 <= 0.2
 0 <= b_k10_BG_s36 <= 0.2
 0 <= b_k10_BG_s37 <= 0.2
 0 <= b_k10_BG_s38 <= 0.2
 0 <= b_k10_BG_s39 <= 0.2
 0 <= b_k10_BG_s40 <= 0.2
 0 <= b_k10_BG_s41 <= 0.2
 0 <= b_k10_BG_s42 <= 0.2
 0 <= b_k10_BG_s43 <= 0.2
 0 <= b_k10_BG_s44 <= 0.2
 0 <= b_k10_BG_s45 <= 0.2
 0 <= b_k10_BG_s46 <= 0.2
 0 <= b_k10_BG_s47 <= 0.2
 0 <= b_k10_BG_s48 <= 0.2
 0 <= b_k10_BG_s49 <= 0.2
 0 <= b_k10_BG_s50 <= 0.2
 0 <= b_k10_BG_s51 <= 0.2
 0 <= b_k10_BG_s52 <= 0.2
 0 <= b_k10_BG_s53 <= 0.2
 0 <= b_k10_BG_s54 <= 0.2
 0 <= b_k10_BG_s55 <= 0.2
 0 <= b_k10_BG_s56 <= 0.2
 0 <= b_k10_BG_s57 <= 0.2
 0 <= b_k10_BG_s58 <= 0.2
 0 <= b_k10_BG_s59 <= 0.2
 0 <= b_k10_BG_s60 <= 0.2
 0 <= b_k10_BG_s61 <= 0.2
 0 <= b_k10_BG_s62 <= 0.2
 0 <= b_k10_BG_s63 <= 0.2
 0 <= b_k10_BG_s64 <= 0.2
 0 <= b_k10_BG_s65 <= 0.2
 0 <= b_k10_BG_s66 <= 0.2
 0 <= b_k10_BG_s67 <= 0.2
 0 <= b_k10_BG_s68 <= 0.2
 0 <= b_k10_BG_s69 <= 0.2
 0 <= b_k10_BG_s70 <= 0.2
 0 <= b_k10_BG_s71 <= 0.2
 0 <= b_k10_BG_s72 <= 0.2
 0 <= b_k10_BG_s73 <= 0.2
 0 <= b_k10_BG_s74 <= 0.2
 0 <= b_k10_BG_s75 <= 0.2
 0 <= b_k10_BG_s76 <= 0.2
 0 <= b_k10_BG_s77 <= 0.2
 0 <= b_k10_BG_s78 <= 0.2
 0 <= b_k10_BG_s79 <= 0.2
 0 <= b_k10_BG_s80 <= 0.2
 0 <= b_k10_BG_s81 <= 0.2
 0 <= b_k10_BG_s82 <= 0.2
 0 <= b_k10_BG_s83 <= 0.2
 0 <= b_k10_BG_s84 <= 0.2
 0 <= b_k10_BG_s85 <= 0.2
 0 <= b_k10_BG_s86 <= 0.2
 0 <= b_k10_BG_s87 <= 0.2
 0 <= b_k10_BG_s88 <= 0.2
 0 <= b_k10_BG_s89 <= 0.2
 0 <= b_k10_BG_s90 <= 0.2
 0 <= b_k10_BG_s91 <= 0.2
 0 <= b_k10_BG_s92 <= 0.2
 0 <= b_k10_BG_s93 <= 0.2
 0 <= b_k10_BG_s94 <= 0.2
 0 <= b_k10_BG_s95 <= 0.2
 0 <= b_k10_BG_s96 <= 0.2
 0 <= b_k10_BG_s97 <= 0.2
 0 <= b_k10_BG_s98 <= 0.2
 0 <= b_k10_BG_s99 <= 0.2
 0 <= b_k10_TSCO_s0 <= 0.2
 0 <= b_k10_TSCO_s1 <= 0.2
 0 <= b_k10_TSCO_s2 <= 0.2
 0 <= b_k10_TSCO_s3 <= 0.2
 0 <= b_k10_TSCO_s4 <= 0.2
 0 <= b_k10_TSCO_s5 <= 0.2
 0 <= b_k10_TSCO_s6 <= 0.2
 0 <= b_k10_TSCO_s7 <= 0.2
 0 <= b_k10_TSCO_s8 <= 0.2
 0 <= b_k10_TSCO_s9 <= 0.2
 0 <= b_k10_TSCO_s00 <= 0.2
 0 <= b_k10_TSCO_s01 <= 0.2
 0 <= b_k10_TSCO_s02 <= 0.2
 0 <= b_k10_TSCO_s03 <= 0.2
 0 <= b_k10_TSCO_s04 <= 0.2
 0 <= b_k10_TSCO_s05 <= 0.2
 0 <= b_k10_TSCO_s06 <= 0.2
 0 <= b_k10_TSCO_s07 <= 0.2
 0 <= b_k10_TSCO_s08 <= 0.2
 0 <= b_k10_TSCO_s09 <= 0.2
 0 <= b_k10_TSCO_s10 <= 0.2
 0 <= b_k10_TSCO_s11 <= 0.2
 0 <= b_k10_TSCO_s12 <= 0.2
 0 <= b_k10_TSCO_s13 <= 0.2
 0 <= b_k10_TSCO_s14 <= 0.2
 0 <= b_k10_TSCO_s15 <= 0.2
 0 <= b_k10_TSCO_s16 <= 0.2
 0 <= b_k10_TSCO_s17 <= 0.2
 0 <= b_k10_TSCO_s18 <= 0.2
 0 <= b_k10_TSCO_s19 <= 0.2
 0 <= b_k10_TSCO_s20 <= 0.2
 0 <= b_k10_TSCO_s21 <= 0.2
 0 <= b_k10_TSCO_s22 <= 0.2
 0 <= b_k10_TSCO_s23 <= 0.2
 0 <= b_k10_TSCO_s24 <= 0.2
 0 <= b_k10_TSCO_s25 <= 0.2
 0 <= b_k10_TSCO_s26 <= 0.2
 0 <= b_k10_TSCO_s27 <= 0.2
 0 <= b_k10_TSCO_s28 <= 0.2
 0 <= b_k10_TSCO_s29 <= 0.2
 0 <= b_k10_TSCO_s30 <= 0.2
 0 <= b_k10_TSCO_s31 <= 0.2
 0 <= b_k10_TSCO_s32 <= 0.2
 0 <= b_k10_TSCO_s33 <= 0.2
 0 <= b_k10_TSCO_s34 <= 0.2
 0 <= b_k10_TSCO_s35 <= 0.2
 0 <= b_k10_TSCO_s36 <= 0.2
 0 <= b_k10_TSCO_s37 <= 0.2
 0 <= b_k10_TSCO_s38 <= 0.2
 0 <= b_k10_TSCO_s39 <= 0.2
 0 <= b_k10_TSCO_s40 <= 0.2
 0 <= b_k10_TSCO_s41 <= 0.2
 0 <= b_k10_TSCO_s42 <= 0.2
 0 <= b_k10_TSCO_s43 <= 0.2
 0 <= b_k10_TSCO_s44 <= 0.2
 0 <= b_k10_TSCO_s45 <= 0.2
 0 <= b_k10_TSCO_s46 <= 0.2
 0 <= b_k10_TSCO_s47 <= 0.2
 0 <= b_k10_TSCO_s48 <= 0.2
 0 <= b_k10_TSCO_s49 <= 0.2
 0 <= b_k10_TSCO_s50 <= 0.2
 0 <= b_k10_TSCO_s51 <= 0.2
 0 <= b_k10_TSCO_s52 <= 0.2
 0 <= b_k10_TSCO_s53 <= 0.2
 0 <= b_k10_TSCO_s54 <= 0.2
 0 <= b_k10_TSCO_s55 <= 0.2
 0 <= b_k10_TSCO_s56 <= 0.2
 0 <= b_k10_TSCO_s57 <= 0.2
 0 <= b_k10_TSCO_s58 <= 0.2
 0 <= b_k10_TSCO_s59 <= 0.2
 0 <= b_k10_TSCO_s60 <= 0.2
 0 <= b_k10_TSCO_s61 <= 0.2
 0 <= b_k10_TSCO_s62 <= 0.2
 0 <= b_k10_TSCO_s63 <= 0.2
 0 <= b_k10_TSCO_s64 <= 0.2
 0 <= b_k10_TSCO_s65 <= 0.2
 0 <= b_k10_TSCO_s66 <= 0.2
 0 <= b_k10_TSCO_s67 <= 0.2
 0 <= b_k10_TSCO_s68 <= 0.2
 0 <= b_k10_TSCO_s69 <= 0.2
 0 <= b_k10_TSCO_s70 <= 0.2
 0 <= b_k10_TSCO_s71 <= 0.2
 0 <= b_k10_TSCO_s72 <= 0.2
 0 <= b_k10_TSCO_s73 <= 0.2
 0 <= b_k10_TSCO_s74 <= 0.2
 0 <= b_k10_TSCO_s75 <= 0.2
 0 <= b_k10_TSCO_s76 <= 0.2
 0 <= b_k10_TSCO_s77 <= 0.2
 0 <= b_k10_TSCO_s78 <= 0.2
 0 <= b_k10_TSCO_s79 <= 0.2
 0 <= b_k10_TSCO_s80 <= 0.2
 0 <= b_k10_TSCO_s81 <= 0.2
 0 <= b_k10_TSCO_s82 <= 0.2
 0 <= b_k10_TSCO_s83 <= 0.2
 0 <= b_k10_TSCO_s84 <= 0.2
 0 <= b_k10_TSCO_s85 <= 0.2
 0 <= b_k10_TSCO_s86 <= 0.2
 0 <= b_k10_TSCO_s87 <= 0.2
 0 <= b_k10_TSCO_s88 <= 0.2
 0 <= b_k10_TSCO_s89 <= 0.2
 0 <= b_k10_TSCO_s90 <= 0.2
 0 <= b_k10_TSCO_s91 <= 0.2
 0 <= b_k10_TSCO_s92 <= 0.2
 0 <= b_k10_TSCO_s93 <= 0.2
 0 <= b_k10_TSCO_s94 <= 0.2
 0 <= b_k10_TSCO_s95 <= 0.2
 0 <= b_k10_TSCO_s96 <= 0.2
 0 <= b_k10_TSCO_s97 <= 0.2
 0 <= b_k10_TSCO_s98 <= 0.2
 0 <= b_k10_TSCO_s99 <= 0.2
 0 <= b_k10_AAL_s0 <= 0.2
 0 <= b_k10_AAL_s1 <= 0.2
 0 <= b_k10_AAL_s2 <= 0.2
 0 <= b_k10_AAL_s3 <= 0.2
 0 <= b_k10_AAL_s4 <= 0.2
 0 <= b_k10_AAL_s5 <= 0.2
 0 <= b_k10_AAL_s6 <= 0.2
 0 <= b_k10_AAL_s7 <= 0.2
 0 <= b_k10_AAL_s8 <= 0.2
 0 <= b_k10_AAL_s9 <= 0.2
 0 <= b_k10_AAL_s00 <= 0.2
 0 <= b_k10_AAL_s01 <= 0.2
 0 <= b_k10_AAL_s02 <= 0.2
 0 <= b_k10_AAL_s03 <= 0.2
 0 <= b_k10_AAL_s04 <= 0.2
 0 <= b_k10_AAL_s05 <= 0.2
 0 <= b_k10_AAL_s06 <= 0.2
 0 <= b_k10_AAL_s07 <= 0.2
 0 <= b_k10_AAL_s08 <= 0.2
 0 <= b_k10_AAL_s09 <= 0.2
 0 <= b_k10_AAL_s10 <= 0.2
 0 <= b_k10_AAL_s11 <= 0.2
 0 <= b_k10_AAL_s12 <= 0.2
 0 <= b_k10_AAL_s13 <= 0.2
 0 <= b_k10_AAL_s14 <= 0.2
 0 <= b_k10_AAL_s15 <= 0.2
 0 <= b_k10_AAL_s16 <= 0.2
 0 <= b_k10_AAL_s17 <= 0.2
 0 <= b_k10_AAL_s18 <= 0.2
 0 <= b_k10_AAL_s19 <= 0.2
 0 <= b_k10_AAL_s20 <= 0.2
 0 <= b_k10_AAL_s21 <= 0.2
 0 <= b_k10_AAL_s22 <= 0.2
 0 <= b_k10_AAL_s23 <= 0.2
 0 <= b_k10_AAL_s24 <= 0.2
 0 <= b_k10_AAL_s25 <= 0.2
 0 <= b_k10_AAL_s26 <= 0.2
 0 <= b_k10_AAL_s27 <= 0.2
 0 <= b_k10_AAL_s28 <= 0.2
 0 <= b_k10_AAL_s29 <= 0.2
 0 <= b_k10_AAL_s30 <= 0.2
 0 <= b_k10_AAL_s31 <= 0.2
 0 <= b_k10_AAL_s32 <= 0.2
 0 <= b_k10_AAL_s33 <= 0.2
 0 <= b_k10_AAL_s34 <= 0.2
 0 <= b_k10_AAL_s35 <= 0.2
 0 <= b_k10_AAL_s36 <= 0.2
 0 <= b_k10_AAL_s37 <= 0.2
 0 <= b_k10_AAL_s38 <= 0.2
 0 <= b_k10_AAL_s39 <= 0.2
 0 <= b_k10_AAL_s40 <= 0.2
 0 <= b_k10_AAL_s41 <= 0.2
 0 <= b_k10_AAL_s42 <= 0.2
 0 <= b_k10_AAL_s43 <= 0.2
 0 <= b_k10_AAL_s44 <= 0.2
 0 <= b_k10_AAL_s45 <= 0.2
 0 <= b_k10_AAL_s46 <= 0.2
 0 <= b_k10_AAL_s47 <= 0.2
 0 <= b_k10_AAL_s48 <= 0.2
 0 <= b_k10_AAL_s49 <= 0.2
 0 <= b_k10_AAL_s50 <= 0.2
 0 <= b_k10_AAL_s51 <= 0.2
 0 <= b_k10_AAL_s52 <= 0.2
 0 <= b_k10_AAL_s53 <= 0.2
 0 <= b_k10_AAL_s54 <= 0.2
 0 <= b_k10_AAL_s55 <= 0.2
 0 <= b_k10_AAL_s56 <= 0.2
 0 <= b_k10_AAL_s57 <= 0.2
 0 <= b_k10_AAL_s58 <= 0.2
 0 <= b_k10_AAL_s59 <= 0.2
 0 <= b_k10_AAL_s60 <= 0.2
 0 <= b_k10_AAL_s61 <= 0.2
 0 <= b_k10_AAL_s62 <= 0.2
 0 <= b_k10_AAL_s63 <= 0.2
 0 <= b_k10_AAL_s64 <= 0.2
 0 <= b_k10_AAL_s65 <= 0.2
 0 <= b_k10_AAL_s66 <= 0.2
 0 <= b_k10_AAL_s67 <= 0.2
 0 <= b_k10_AAL_s68 <= 0.2
 0 <= b_k10_AAL_s69 <= 0.2
 0 <= b_k10_AAL_s70 <= 0.2
 0 <= b_k10_AAL_s71 <= 0.2
 0 <= b_k10_AAL_s72 <= 0.2
 0 <= b_k10_AAL_s73 <= 0.2
 0 <= b_k10_AAL_s74 <= 0.2
 0 <= b_k10_AAL_s75 <= 0.2
 0 <= b_k10_AAL_s76 <= 0.2
 0 <= b_k10_AAL_s77 <= 0.2
 0 <= b_k10_AAL_s78 <= 0.2
 0 <= b_k10_AAL_s79 <= 0.2
 0 <= b_k10_AAL_s80 <= 0.2
 0 <= b_k10_AAL_s81 <= 0.2
 0 <= b_k10_AAL_s82 <= 0.2
 0 <= b_k10_AAL_s83 <= 0.2
 0 <= b_k10_AAL_s84 <= 0.2
 0 <= b_k10_AAL_s85 <= 0.2
 0 <= b_k10_AAL_s86 <= 0.2
 0 <= b_k10_AAL_s87 <= 0.2
 0 <= b_k10_AAL_s88 <= 0.2
 0 <= b_k10_AAL_s89 <= 0.2
 0 <= b_k10_AAL_s90 <= 0.2
 0 <= b_k10_AAL_s91 <= 0.2
 0 <= b_k10_AAL_s92 <= 0.2
 0 <= b_k10_AAL_s93 <= 0.2
 0 <= b_k10_AAL_s94 <= 0.2
 0 <= b_k10_AAL_s95 <= 0.2
 0 <= b_k10_AAL_s96 <= 0.2
 0 <= b_k10_AAL_s97 <= 0.2
 0 <= b_k10_AAL_s98 <= 0.2
 0 <= b_k10_AAL_s99 <= 0.2
 0 <= b_k10_PRU_s0 <= 0.2
 0 <= b_k10_PRU_s1 <= 0.2
 0 <= b_k10_PRU_s2 <= 0.2
 0 <= b_k10_PRU_s3 <= 0.2
 0 <= b_k10_PRU_s4 <= 0.2
 0 <= b_k10_PRU_s5 <= 0.2
 0 <= b_k10_PRU_s6 <= 0.2
 0 <= b_k10_PRU_s7 <= 0.2
 0 <= b_k10_PRU_s8 <= 0.2
 0 <= b_k10_PRU_s9 <= 0.2
 0 <= b_k10_PRU_s00 <= 0.2
 0 <= b_k10_PRU_s01 <= 0.2
 0 <= b_k10_PRU_s02 <= 0.2
 0 <= b_k10_PRU_s03 <= 0.2
 0 <= b_k10_PRU_s04 <= 0.2
 0 <= b_k10_PRU_s05 <= 0.2
 0 <= b_k10_PRU_s06 <= 0.2
 0 <= b_k10_PRU_s07 <= 0.2
 0 <= b_k10_PRU_s08 <= 0.2
 0 <= b_k10_PRU_s09 <= 0.2
 0 <= b_k10_PRU_s10 <= 0.2
 0 <= b_k10_PRU_s11 <= 0.2
 0 <= b_k10_PRU_s12 <= 0.2
 0 <= b_k10_PRU_s13 <= 0.2
 0 <= b_k10_PRU_s14 <= 0.2
 0 <= b_k10_PRU_s15 <= 0.2
 0 <= b_k10_PRU_s16 <= 0.2
 0 <= b_k10_PRU_s17 <= 0.2
 0 <= b_k10_PRU_s18 <= 0.2
 0 <= b_k10_PRU_s19 <= 0.2
 0 <= b_k10_PRU_s20 <= 0.2
 0 <= b_k10_PRU_s21 <= 0.2
 0 <= b_k10_PRU_s22 <= 0.2
 0 <= b_k10_PRU_s23 <= 0.2
 0 <= b_k10_PRU_s24 <= 0.2
 0 <= b_k10_PRU_s25 <= 0.2
 0 <= b_k10_PRU_s26 <= 0.2
 0 <= b_k10_PRU_s27 <= 0.2
 0 <= b_k10_PRU_s28 <= 0.2
 0 <= b_k10_PRU_s29 <= 0.2
 0 <= b_k10_PRU_s30 <= 0.2
 0 <= b_k10_PRU_s31 <= 0.2
 0 <= b_k10_PRU_s32 <= 0.2
 0 <= b_k10_PRU_s33 <= 0.2
 0 <= b_k10_PRU_s34 <= 0.2
 0 <= b_k10_PRU_s35 <= 0.2
 0 <= b_k10_PRU_s36 <= 0.2
 0 <= b_k10_PRU_s37 <= 0.2
 0 <= b_k10_PRU_s38 <= 0.2
 0 <= b_k10_PRU_s39 <= 0.2
 0 <= b_k10_PRU_s40 <= 0.2
 0 <= b_k10_PRU_s41 <= 0.2
 0 <= b_k10_PRU_s42 <= 0.2
 0 <= b_k10_PRU_s43 <= 0.2
 0 <= b_k10_PRU_s44 <= 0.2
 0 <= b_k10_PRU_s45 <= 0.2
 0 <= b_k10_PRU_s46 <= 0.2
 0 <= b_k10_PRU_s47 <= 0.2
 0 <= b_k10_PRU_s48 <= 0.2
 0 <= b_k10_PRU_s49 <= 0.2
 0 <= b_k10_PRU_s50 <= 0.2
 0 <= b_k10_PRU_s51 <= 0.2
 0 <= b_k10_PRU_s52 <= 0.2
 0 <= b_k10_PRU_s53 <= 0.2
 0 <= b_k10_PRU_s54 <= 0.2
 0 <= b_k10_PRU_s55 <= 0.2
 0 <= b_k10_PRU_s56 <= 0.2
 0 <= b_k10_PRU_s57 <= 0.2
 0 <= b_k10_PRU_s58 <= 0.2
 0 <= b_k10_PRU_s59 <= 0.2
 0 <= b_k10_PRU_s60 <= 0.2
 0 <= b_k10_PRU_s61 <= 0.2
 0 <= b_k10_PRU_s62 <= 0.2
 0 <= b_k10_PRU_s63 <= 0.2
 0 <= b_k10_PRU_s64 <= 0.2
 0 <= b_k10_PRU_s65 <= 0.2
 0 <= b_k10_PRU_s66 <= 0.2
 0 <= b_k10_PRU_s67 <= 0.2
 0 <= b_k10_PRU_s68 <= 0.2
 0 <= b_k10_PRU_s69 <= 0.2
 0 <= b_k10_PRU_s70 <= 0.2
 0 <= b_k10_PRU_s71 <= 0.2
 0 <= b_k10_PRU_s72 <= 0.2
 0 <= b_k10_PRU_s73 <= 0.2
 0 <= b_k10_PRU_s74 <= 0.2
 0 <= b_k10_PRU_s75 <= 0.2
 0 <= b_k10_PRU_s76 <= 0.2
 0 <= b_k10_PRU_s77 <= 0.2
 0 <= b_k10_PRU_s78 <= 0.2
 0 <= b_k10_PRU_s79 <= 0.2
 0 <= b_k10_PRU_s80 <= 0.2
 0 <= b_k10_PRU_s81 <= 0.2
 0 <= b_k10_PRU_s82 <= 0.2
 0 <= b_k10_PRU_s83 <= 0.2
 0 <= b_k10_PRU_s84 <= 0.2
 0 <= b_k10_PRU_s85 <= 0.2
 0 <= b_k10_PRU_s86 <= 0.2
 0 <= b_k10_PRU_s87 <= 0.2
 0 <= b_k10_PRU_s88 <= 0.2
 0 <= b_k10_PRU_s89 <= 0.2
 0 <= b_k10_PRU_s90 <= 0.2
 0 <= b_k10_PRU_s91 <= 0.2
 0 <= b_k10_PRU_s92 <= 0.2
 0 <= b_k10_PRU_s93 <= 0.2
 0 <= b_k10_PRU_s94 <= 0.2
 0 <= b_k10_PRU_s95 <= 0.2
 0 <= b_k10_PRU_s96 <= 0.2
 0 <= b_k10_PRU_s97 <= 0.2
 0 <= b_k10_PRU_s98 <= 0.2
 0 <= b_k10_PRU_s99 <= 0.2
 0 <= s_0_HSBC <= 0.2
 0 <= s_0_VOD <= 0.2
 0 <= s_0_BP <= 0.2
 0 <= s_0_GSK <= 0.2
 0 <= s_0_AZN <= 0.2
 0 <= s_0_RIO <= 0.2
 0 <= s_0_BG <= 0.2
 0 <= s_0_TSCO <= 0.2
 0 <= s_0_AAL <= 0.2
 0 <= s_0_PRU <= 0.2
 0 <= s_k10_HSBC_s0 <= 0.2
 0 <= s_k10_HSBC_s1 <= 0.2
 0 <= s_k10_HSBC_s2 <= 0.2
 0 <= s_k10_HSBC_s3 <= 0.2
 0 <= s_k10_HSBC_s4 <= 0.2
 0 <= s_k10_HSBC_s5 <= 0.2
 0 <= s_k10_HSBC_s6 <= 0.2
 0 <= s_k10_HSBC_s7 <= 0.2
 0 <= s_k10_HSBC_s8 <= 0.2
 0 <= s_k10_HSBC_s9 <= 0.2
 0 <= s_k10_HSBC_s00 <= 0.2
 0 <= s_k10_HSBC_s01 <= 0.2
 0 <= s_k10_HSBC_s02 <= 0.2
 0 <= s_k10_HSBC_s03 <= 0.2
 0 <= s_k10_HSBC_s04 <= 0.2
 0 <= s_k10_HSBC_s05 <= 0.2
 0 <= s_k10_HSBC_s06 <= 0.2
 0 <= s_k10_HSBC_s07 <= 0.2
 0 <= s_k10_HSBC_s08 <= 0.2
 0 <= s_k10_HSBC_s09 <= 0.2
 0 <= s_k10_HSBC_s10 <= 0.2
 0 <= s_k10_HSBC_s11 <= 0.2
 0 <= s_k10_HSBC_s12 <= 0.2
 0 <= s_k10_HSBC_s13 <= 0.2
 0 <= s_k10_HSBC_s14 <= 0.2
 0 <= s_k10_HSBC_s15 <= 0.2
 0 <= s_k10_HSBC_s16 <= 0.2
 0 <= s_k10_HSBC_s17 <= 0.2
 0 <= s_k10_HSBC_s18 <= 0.2
 0 <= s_k10_HSBC_s19 <= 0.2
 0 <= s_k10_HSBC_s20 <= 0.2
 0 <= s_k10_HSBC_s21 <= 0.2
 0 <= s_k10_HSBC_s22 <= 0.2
 0 <= s_k10_HSBC_s23 <= 0.2
 0 <= s_k10_HSBC_s24 <= 0.2
 0 <= s_k10_HSBC_s25 <= 0.2
 0 <= s_k10_HSBC_s26 <= 0.2
 0 <= s_k10_HSBC_s27 <= 0.2
 0 <= s_k10_HSBC_s28 <= 0.2
 0 <= s_k10_HSBC_s29 <= 0.2
 0 <= s_k10_HSBC_s30 <= 0.2
 0 <= s_k10_HSBC_s31 <= 0.2
 0 <= s_k10_HSBC_s32 <= 0.2
 0 <= s_k10_HSBC_s33 <= 0.2
 0 <= s_k10_HSBC_s34 <= 0.2
 0 <= s_k10_HSBC_s35 <= 0.2
 0 <= s_k10_HSBC_s36 <= 0.2
 0 <= s_k10_HSBC_s37 <= 0.2
 0 <= s_k10_HSBC_s38 <= 0.2
 0 <= s_k10_HSBC_s39 <= 0.2
 0 <= s_k10_HSBC_s40 <= 0.2
 0 <= s_k10_HSBC_s41 <= 0.2
 0 <= s_k10_HSBC_s42 <= 0.2
 0 <= s_k10_HSBC_s43 <= 0.2
 0 <= s_k10_HSBC_s44 <= 0.2
 0 <= s_k10_HSBC_s45 <= 0.2
 0 <= s_k10_HSBC_s46 <= 0.2
 0 <= s_k10_HSBC_s47 <= 0.2
 0 <= s_k10_HSBC_s48 <= 0.2
 0 <= s_k10_HSBC_s49 <= 0.2
 0 <= s_k10_HSBC_s50 <= 0.2
 0 <= s_k10_HSBC_s51 <= 0.2
 0 <= s_k10_HSBC_s52 <= 0.2
 0 <= s_k10_HSBC_s53 <= 0.2
 0 <= s_k10_HSBC_s54 <= 0.2
 0 <= s_k10_HSBC_s55 <= 0.2
 0 <= s_k10_HSBC_s56 <= 0.2
 0 <= s_k10_HSBC_s57 <= 0.2
 0 <= s_k10_HSBC_s58 <= 0.2
 0 <= s_k10_HSBC_s59 <= 0.2
 0 <= s_k10_HSBC_s60 <= 0.2
 0 <= s_k10_HSBC_s61 <= 0.2
 0 <= s_k10_HSBC_s62 <= 0.2
 0 <= s_k10_HSBC_s63 <= 0.2
 0 <= s_k10_HSBC_s64 <= 0.2
 0 <= s_k10_HSBC_s65 <= 0.2
 0 <= s_k10_HSBC_s66 <= 0.2
 0 <= s_k10_HSBC_s67 <= 0.2
 0 <= s_k10_HSBC_s68 <= 0.2
 0 <= s_k10_HSBC_s69 <= 0.2
 0 <= s_k10_HSBC_s70 <= 0.2
 0 <= s_k10_HSBC_s71 <= 0.2
 0 <= s_k10_HSBC_s72 <= 0.2
 0 <= s_k10_HSBC_s73 <= 0.2
 0 <= s_k10_HSBC_s74 <= 0.2
 0 <= s_k10_HSBC_s75 <= 0.2
 0 <= s_k10_HSBC_s76 <= 0.2
 0 <= s_k10_HSBC_s77 <= 0.2
 0 <= s_k10_HSBC_s78 <= 0.2
 0 <= s_k10_HSBC_s79 <= 0.2
 0 <= s_k10_HSBC_s80 <= 0.2
 0 <= s_k10_HSBC_s81 <= 0.2
 0 <= s_k10_HSBC_s82 <= 0.2
 0 <= s_k10_HSBC_s83 <= 0.2
 0 <= s_k10_HSBC_s84 <= 0.2
 0 <= s_k10_HSBC_s85 <= 0.2
 0 <= s_k10_HSBC_s86 <= 0.2
 0 <= s_k10_HSBC_s87 <= 0.2
 0 <= s_k10_HSBC_s88 <= 0.2
 0 <= s_k10_HSBC_s89 <= 0.2
 0 <= s_k10_HSBC_s90 <= 0.2
 0 <= s_k10_HSBC_s91 <= 0.2
 0 <= s_k10_HSBC_s92 <= 0.2
 0 <= s_k10_HSBC_s93 <= 0.2
 0 <= s_k10_HSBC_s94 <= 0.2
 0 <= s_k10_HSBC_s95 <= 0.2
 0 <= s_k10_HSBC_s96 <= 0.2
 0 <= s_k10_HSBC_s97 <= 0.2
 0 <= s_k10_HSBC_s98 <= 0.2
 0 <= s_k10_HSBC_s99 <= 0.2
 0 <= s_k10_VOD_s0 <= 0.2
 0 <= s_k10_VOD_s1 <= 0.2
 0 <= s_k10_VOD_s2 <= 0.2
 0 <= s_k10_VOD_s3 <= 0.2
 0 <= s_k10_VOD_s4 <= 0.2
 0 <= s_k10_VOD_s5 <= 0.2
 0 <= s_k10_VOD_s6 <= 0.2
 0 <= s_k10_VOD_s7 <= 0.2
 0 <= s_k10_VOD_s8 <= 0.2
 0 <= s_k10_VOD_s9 <= 0.2
 0 <= s_k10_VOD_s00 <= 0.2
 0 <= s_k10_VOD_s01 <= 0.2
 0 <= s_k10_VOD_s02 <= 0.2
 0 <= s_k10_VOD_s03 <= 0.2
 0 <= s_k10_VOD_s04 <= 0.2
 0 <= s_k10_VOD_s05 <= 0.2
 0 <= s_k10_VOD_s06 <= 0.2
 0 <= s_k10_VOD_s07 <= 0.2
 0 <= s_k10_VOD_s08 <= 0.2
 0 <= s_k10_VOD_s09 <= 0.2
 0 <= s_k10_VOD_s10 <= 0.2
 0 <= s_k10_VOD_s11 <= 0.2
 0 <= s_k10_VOD_s12 <= 0.2
 0 <= s_k10_VOD_s13 <= 0.2
 0 <= s_k10_VOD_s14 <= 0.2
 0 <= s_k10_VOD_s15 <= 0.2
 0 <= s_k10_VOD_s16 <= 0.2
 0 <= s_k10_VOD_s17 <= 0.2
 0 <= s_k10_VOD_s18 <= 0.2
 0 <= s_k10_VOD_s19 <= 0.2
 0 <= s_k10_VOD_s20 <= 0.2
 0 <= s_k10_VOD_s21 <= 0.2
 0 <= s_k10_VOD_s22 <= 0.2
 0 <= s_k10_VOD_s23 <= 0.2
 0 <= s_k10_VOD_s24 <= 0.2
 0 <= s_k10_VOD_s25 <= 0.2
 0 <= s_k10_VOD_s26 <= 0.2
 0 <= s_k10_VOD_s27 <= 0.2
 0 <= s_k10_VOD_s28 <= 0.2
 0 <= s_k10_VOD_s29 <= 0.2
 0 <= s_k10_VOD_s30 <= 0.2
 0 <= s_k10_VOD_s31 <= 0.2
 0 <= s_k10_VOD_s32 <= 0.2
 0 <= s_k10_VOD_s33 <= 0.2
 0 <= s_k10_VOD_s34 <= 0.2
 0 <= s_k10_VOD_s35 <= 0.2
 0 <= s_k10_VOD_s36 <= 0.2
 0 <= s_k10_VOD_s37 <= 0.2
 0 <= s_k10_VOD_s38 <= 0.2
 0 <= s_k10_VOD_s39 <= 0.2
 0 <= s_k10_VOD_s40 <= 0.2
 0 <= s_k10_VOD_s41 <= 0.2
 0 <= s_k10_VOD_s42 <= 0.2
 0 <= s_k10_VOD_s43 <= 0.2
 0 <= s_k10_VOD_s44 <= 0.2
 0 <= s_k10_VOD_s45 <= 0.2
 0 <= s_k10_VOD_s46 <= 0.2
 0 <= s_k10_VOD_s47 <= 0.2
 0 <= s_k10_VOD_s48 <= 0.2
 0 <= s_k10_VOD_s49 <= 0.2
 0 <= s_k10_VOD_s50 <= 0.2
 0 <= s_k10_VOD_s51 <= 0.2
 0 <= s_k10_VOD_s52 <= 0.2
 0 <= s_k10_VOD_s53 <= 0.2
 0 <= s_k10_VOD_s54 <= 0.2
 0 <= s_k10_VOD_s55 <= 0.2
 0 <= s_k10_VOD_s56 <= 0.2
 0 <= s_k10_VOD_s57 <= 0.2
 0 <= s_k10_VOD_s58 <= 0.2
 0 <= s_k10_VOD_s59 <= 0.2
 0 <= s_k10_VOD_s60 <= 0.2
 0 <= s_k10_VOD_s61 <= 0.2
 0 <= s_k10_VOD_s62 <= 0.2
 0 <= s_k10_VOD_s63 <= 0.2
 0 <= s_k10_VOD_s64 <= 0.2
 0 <= s_k10_VOD_s65 <= 0.2
 0 <= s_k10_VOD_s66 <= 0.2
 0 <= s_k10_VOD_s67 <= 0.2
 0 <= s_k10_VOD_s68 <= 0.2
 0 <= s_k10_VOD_s69 <= 0.2
 0 <= s_k10_VOD_s70 <= 0.2
 0 <= s_k10_VOD_s71 <= 0.2
 0 <= s_k10_VOD_s72 <= 0.2
 0 <= s_k10_VOD_s73 <= 0.2
 0 <= s_k10_VOD_s74 <= 0.2
 0 <= s_k10_VOD_s75 <= 0.2
 0 <= s_k10_VOD_s76 <= 0.2
 0 <= s_k10_VOD_s77 <= 0.2
 0 <= s_k10_VOD_s78 <= 0.2
 0 <= s_k10_VOD_s79 <= 0.2
 0 <= s_k10_VOD_s80 <= 0.2
 0 <= s_k10_VOD_s81 <= 0.2
 0 <= s_k10_VOD_s82 <= 0.2
 0 <= s_k10_VOD_s83 <= 0.2
 0 <= s_k10_VOD_s84 <= 0.2
 0 <= s_k10_VOD_s85 <= 0.2
 0 <= s_k10_VOD_s86 <= 0.2
 0 <= s_k10_VOD_s87 <= 0.2
 0 <= s_k10_VOD_s88 <= 0.2
 0 <= s_k10_VOD_s89 <= 0.2
 0 <= s_k10_VOD_s90 <= 0.2
 0 <= s_k10_VOD_s91 <= 0.2
 0 <= s_k10_VOD_s92 <= 0.2
 0 <= s_k10_VOD_s93 <= 0.2
 0 <= s_k10_VOD_s94 <= 0.2
 0 <= s_k10_VOD_s95 <= 0.2
 0 <= s_k10_VOD_s96 <= 0.2
 0 <= s_k10_VOD_s97 <= 0.2
 0 <= s_k10_VOD_s98 <= 0.2
 0 <= s_k10_VOD_s99 <= 0.2
 0 <= s_k10_BP_s0 <= 0.2
 0 <= s_k10_BP_s1 <= 0.2
 0 <= s_k10_BP_s2 <= 0.2
 0 <= s_k10_BP_s3 <= 0.2
 0 <= s_k10_BP_s4 <= 0.2
 0 <= s_k10_BP_s5 <= 0.2
 0 <= s_k10_BP_s6 <= 0.2
 0 <= s_k10_BP_s7 <= 0.2
 0 <= s_k10_BP_s8 <= 0.2
 0 <= s_k10_BP_s9 <= 0.2
 0 <= s_k10_BP_s00 <= 0.2
 0 <= s_k10_BP_s01 <= 0.2
 0 <= s_k10_BP_s02 <= 0.2
 0 <= s_k10_BP_s03 <= 0.2
 0 <= s_k10_BP_s04 <= 0.2
 0 <= s_k10_BP_s05 <= 0.2
 0 <= s_k10_BP_s06 <= 0.2
 0 <= s_k10_BP_s07 <= 0.2
 0 <= s_k10_BP_s08 <= 0.2
 0 <= s_k10_BP_s09 <= 0.2
 0 <= s_k10_BP_s10 <= 0.2
 0 <= s_k10_BP_s11 <= 0.2
 0 <= s_k10_BP_s12 <= 0.2
 0 <= s_k10_BP_s13 <= 0.2
 0 <= s_k10_BP_s14 <= 0.2
 0 <= s_k10_BP_s15 <= 0.2
 0 <= s_k10_BP_s16 <= 0.2
 0 <= s_k10_BP_s17 <= 0.2
 0 <= s_k10_BP_s18 <= 0.2
 0 <= s_k10_BP_s19 <= 0.2
 0 <= s_k10_BP_s20 <= 0.2
 0 <= s_k10_BP_s21 <= 0.2
 0 <= s_k10_BP_s22 <= 0.2
 0 <= s_k10_BP_s23 <= 0.2
 0 <= s_k10_BP_s24 <= 0.2
 0 <= s_k10_BP_s25 <= 0.2
 0 <= s_k10_BP_s26 <= 0.2
 0 <= s_k10_BP_s27 <= 0.2
 0 <= s_k10_BP_s28 <= 0.2
 0 <= s_k10_BP_s29 <= 0.2
 0 <= s_k10_BP_s30 <= 0.2
 0 <= s_k10_BP_s31 <= 0.2
 0 <= s_k10_BP_s32 <= 0.2
 0 <= s_k10_BP_s33 <= 0.2
 0 <= s_k10_BP_s34 <= 0.2
 0 <= s_k10_BP_s35 <= 0.2
 0 <= s_k10_BP_s36 <= 0.2
 0 <= s_k10_BP_s37 <= 0.2
 0 <= s_k10_BP_s38 <= 0.2
 0 <= s_k10_BP_s39 <= 0.2
 0 <= s_k10_BP_s40 <= 0.2
 0 <= s_k10_BP_s41 <= 0.2
 0 <= s_k10_BP_s42 <= 0.2
 0 <= s_k10_BP_s43 <= 0.2
 0 <= s_k10_BP_s44 <= 0.2
 0 <= s_k10_BP_s45 <= 0.2
 0 <= s_k10_BP_s46 <= 0.2
 0 <= s_k10_BP_s47 <= 0.2
 0 <= s_k10_BP_s48 <= 0.2
 0 <= s_k10_BP_s49 <= 0.2
 0 <= s_k10_BP_s50 <= 0.2
 0 <= s_k10_BP_s51 <= 0.2
 0 <= s_k10_BP_s52 <= 0.2
 0 <= s_k10_BP_s53 <= 0.2
 0 <= s_k10_BP_s54 <= 0.2
 0 <= s_k10_BP_s55 <= 0.2
 0 <= s_k10_BP_s56 <= 0.2
 0 <= s_k10_BP_s57 <= 0.2
 0 <= s_k10_BP_s58 <= 0.2
 0 <= s_k10_BP_s59 <= 0.2
 0 <= s_k10_BP_s60 <= 0.2
 0 <= s_k10_BP_s61 <= 0.2
 0 <= s_k10_BP_s62 <= 0.2
 0 <= s_k10_BP_s63 <= 0.2
 0 <= s_k10_BP_s64 <= 0.2
 0 <= s_k10_BP_s65 <= 0.2
 0 <= s_k10_BP_s66 <= 0.2
 0 <= s_k10_BP_s67 <= 0.2
 0 <= s_k10_BP_s68 <= 0.2
 0 <= s_k10_BP_s69 <= 0.2
 0 <= s_k10_BP_s70 <= 0.2
 0 <= s_k10_BP_s71 <= 0.2
 0 <= s_k10_BP_s72 <= 0.2
 0 <= s_k10_BP_s73 <= 0.2
 0 <= s_k10_BP_s74 <= 0.2
 0 <= s_k10_BP_s75 <= 0.2
 0 <= s_k10_BP_s76 <= 0.2
 0 <= s_k10_BP_s77 <= 0.2
 0 <= s_k10_BP_s78 <= 0.2
 0 <= s_k10_BP_s79 <= 0.2
 0 <= s_k10_BP_s80 <= 0.2
 0 <= s_k10_BP_s81 <= 0.2
 0 <= s_k10_BP_s82 <= 0.2
 0 <= s_k10_BP_s83 <= 0.2
 0 <= s_k10_BP_s84 <= 0.2
 0 <= s_k10_BP_s85 <= 0.2
 0 <= s_k10_BP_s86 <= 0.2
 0 <= s_k10_BP_s87 <= 0.2
 0 <= s_k10_BP_s88 <= 0.2
 0 <= s_k10_BP_s89 <= 0.2
 0 <= s_k10_BP_s90 <= 0.2
 0 <= s_k10_BP_s91 <= 0.2
 0 <= s_k10_BP_s92 <= 0.2
 0 <= s_k10_BP_s93 <= 0.2
 0 <= s_k10_BP_s94 <= 0.2
 0 <= s_k10_BP_s95 <= 0.2
 0 <= s_k10_BP_s96 <= 0.2
 0 <= s_k10_BP_s97 <= 0.2
 0 <= s_k10_BP_s98 <= 0.2
 0 <= s_k10_BP_s99 <= 0.2
 0 <= s_k10_GSK_s0 <= 0.2
 0 <= s_k10_GSK_s1 <= 0.2
 0 <= s_k10_GSK_s2 <= 0.2
 0 <= s_k10_GSK_s3 <= 0.2
 0 <= s_k10_GSK_s4 <= 0.2
 0 <= s_k10_GSK_s5 <= 0.2
 0 <= s_k10_GSK_s6 <= 0.2
 0 <= s_k10_GSK_s7 <= 0.2
 0 <= s_k10_GSK_s8 <= 0.2
 0 <= s_k10_GSK_s9 <= 0.2
 0 <= s_k10_GSK_s00 <= 0.2
 0 <= s_k10_GSK_s01 <= 0.2
 0 <= s_k10_GSK_s02 <= 0.2
 0 <= s_k10_GSK_s03 <= 0.2
 0 <= s_k10_GSK_s04 <= 0.2
 0 <= s_k10_GSK_s05 <= 0.2
 0 <= s_k10_GSK_s06 <= 0.2
 0 <= s_k10_GSK_s07 <= 0.2
 0 <= s_k10_GSK_s08 <= 0.2
 0 <= s_k10_GSK_s09 <= 0.2
 0 <= s_k10_GSK_s10 <= 0.2
 0 <= s_k10_GSK_s11 <= 0.2
 0 <= s_k10_GSK_s12 <= 0.2
 0 <= s_k10_GSK_s13 <= 0.2
 0 <= s_k10_GSK_s14 <= 0.2
 0 <= s_k10_GSK_s15 <= 0.2
 0 <= s_k10_GSK_s16 <= 0.2
 0 <= s_k10_GSK_s17 <= 0.2
 0 <= s_k10_GSK_s18 <= 0.2
 0 <= s_k10_GSK_s19 <= 0.2
 0 <= s_k10_GSK_s20 <= 0.2
 0 <= s_k10_GSK_s21 <= 0.2
 0 <= s_k10_GSK_s22 <= 0.2
 0 <= s_k10_GSK_s23 <= 0.2
 0 <= s_k10_GSK_s24 <= 0.2
 0 <= s_k10_GSK_s25 <= 0.2
 0 <= s_k10_GSK_s26 <= 0.2
 0 <= s_k10_GSK_s27 <= 0.2
 0 <= s_k10_GSK_s28 <= 0.2
 0 <= s_k10_GSK_s29 <= 0.2
 0 <= s_k10_GSK_s30 <= 0.2
 0 <= s_k10_GSK_s31 <= 0.2
 0 <= s_k10_GSK_s32 <= 0.2
 0 <= s_k10_GSK_s33 <= 0.2
 0 <= s_k10_GSK_s34 <= 0.2
 0 <= s_k10_GSK_s35 <= 0.2
 0 <= s_k10_GSK_s36 <= 0.2
 0 <= s_k10_GSK_s37 <= 0.2
 0 <= s_k10_GSK_s38 <= 0.2
 0 <= s_k10_GSK_s39 <= 0.2
 0 <= s_k10_GSK_s40 <= 0.2
 0 <= s_k10_GSK_s41 <= 0.2
 0 <= s_k10_GSK_s42 <= 0.2
 0 <= s_k10_GSK_s43 <= 0.2
 0 <= s_k10_GSK_s44 <= 0.2
 0 <= s_k10_GSK_s45 <= 0.2
 0 <= s_k10_GSK_s46 <= 0.2
 0 <= s_k10_GSK_s47 <= 0.2
 0 <= s_k10_GSK_s48 <= 0.2
 0 <= s_k10_GSK_s49 <= 0.2
 0 <= s_k10_GSK_s50 <= 0.2
 0 <= s_k10_GSK_s51 <= 0.2
 0 <= s_k10_GSK_s52 <= 0.2
 0 <= s_k10_GSK_s53 <= 0.2
 0 <= s_k10_GSK_s54 <= 0.2
 0 <= s_k10_GSK_s55 <= 0.2
 0 <= s_k10_GSK_s56 <= 0.2
 0 <= s_k10_GSK_s57 <= 0.2
 0 <= s_k10_GSK_s58 <= 0.2
 0 <= s_k10_GSK_s59 <= 0.2
 0 <= s_k10_GSK_s60 <= 0.2
 0 <= s_k10_GSK_s61 <= 0.2
 0 <= s_k10_GSK_s62 <= 0.2
 0 <= s_k10_GSK_s63 <= 0.2
 0 <= s_k10_GSK_s64 <= 0.2
 0 <= s_k10_GSK_s65 <= 0.2
 0 <= s_k10_GSK_s66 <= 0.2
 0 <= s_k10_GSK_s67 <= 0.2
 0 <= s_k10_GSK_s68 <= 0.2
 0 <= s_k10_GSK_s69 <= 0.2
 0 <= s_k10_GSK_s70 <= 0.2
 0 <= s_k10_GSK_s71 <= 0.2
 0 <= s_k10_GSK_s72 <= 0.2
 0 <= s_k10_GSK_s73 <= 0.2
 0 <= s_k10_GSK_s74 <= 0.2
 0 <= s_k10_GSK_s75 <= 0.2
 0 <= s_k10_GSK_s76 <= 0.2
 0 <= s_k10_GSK_s77 <= 0.2
 0 <= s_k10_GSK_s78 <= 0.2
 0 <= s_k10_GSK_s79 <= 0.2
 0 <= s_k10_GSK_s80 <= 0.2
 0 <= s_k10_GSK_s81 <= 0.2
 0 <= s_k10_GSK_s82 <= 0.2
 0 <= s_k10_GSK_s83 <= 0.2
 0 <= s_k10_GSK_s84 <= 0.2
 0 <= s_k10_GSK_s85 <= 0.2
 0 <= s_k10_GSK_s86 <= 0.2
 0 <= s_k10_GSK_s87 <= 0.2
 0 <= s_k10_GSK_s88 <= 0.2
 0 <= s_k10_GSK_s89 <= 0.2
 0 <= s_k10_GSK_s90 <= 0.2
 0 <= s_k10_GSK_s91 <= 0.2
 0 <= s_k10_GSK_s92 <= 0.2
 0 <= s_k10_GSK_s93 <= 0.2
 0 <= s_k10_GSK_s94 <= 0.2
 0 <= s_k10_GSK_s95 <= 0.2
 0 <= s_k10_GSK_s96 <= 0.2
 0 <= s_k10_GSK_s97 <= 0.2
 0 <= s_k10_GSK_s98 <= 0.2
 0 <= s_k10_GSK_s99 <= 0.2
 0 <= s_k10_AZN_s0 <= 0.2
 0 <= s_k10_AZN_s1 <= 0.2
 0 <= s_k10_AZN_s2 <= 0.2
 0 <= s_k10_AZN_s3 <= 0.2
 0 <= s_k10_AZN_s4 <= 0.2
 0 <= s_k10_AZN_s5 <= 0.2
 0 <= s_k10_AZN_s6 <= 0.2
 0 <= s_k10_AZN_s7 <= 0.2
 0 <= s_k10_AZN_s8 <= 0.2
 0 <= s_k10_AZN_s9 <= 0.2
 0 <= s_k10_AZN_s00 <= 0.2
 0 <= s_k10_AZN_s01 <= 0.2
 0 <= s_k10_AZN_s02 <= 0.2
 0 <= s_k10_AZN_s03 <= 0.2
 0 <= s_k10_AZN_s04 <= 0.2
 0 <= s_k10_AZN_s05 <= 0.2
 0 <= s_k10_AZN_s06 <= 0.2
 0 <= s_k10_AZN_s07 <= 0.2
 0 <= s_k10_AZN_s08 <= 0.2
 0 <= s_k10_AZN_s09 <= 0.2
 0 <= s_k10_AZN_s10 <= 0.2
 0 <= s_k10_AZN_s11 <= 0.2
 0 <= s_k10_AZN_s12 <= 0.2
 0 <= s_k10_AZN_s13 <= 0.2
 0 <= s_k10_AZN_s14 <= 0.2
 0 <= s_k10_AZN_s15 <= 0.2
 0 <= s_k10_AZN_s16 <= 0.2
 0 <= s_k10_AZN_s17 <= 0.2
 0 <= s_k10_AZN_s18 <= 0.2
 0 <= s_k10_AZN_s19 <= 0.2
 0 <= s_k10_AZN_s20 <= 0.2
 0 <= s_k10_AZN_s21 <= 0.2
 0 <= s_k10_AZN_s22 <= 0.2
 0 <= s_k10_AZN_s23 <= 0.2
 0 <= s_k10_AZN_s24 <= 0.2
 0 <= s_k10_AZN_s25 <= 0.2
 0 <= s_k10_AZN_s26 <= 0.2
 0 <= s_k10_AZN_s27 <= 0.2
 0 <= s_k10_AZN_s28 <= 0.2
 0 <= s_k10_AZN_s29 <= 0.2
 0 <= s_k10_AZN_s30 <= 0.2
 0 <= s_k10_AZN_s31 <= 0.2
 0 <= s_k10_AZN_s32 <= 0.2
 0 <= s_k10_AZN_s33 <= 0.2
 0 <= s_k10_AZN_s34 <= 0.2
 0 <= s_k10_AZN_s35 <= 0.2
 0 <= s_k10_AZN_s36 <= 0.2
 0 <= s_k10_AZN_s37 <= 0.2
 0 <= s_k10_AZN_s38 <= 0.2
 0 <= s_k10_AZN_s39 <= 0.2
 0 <= s_k10_AZN_s40 <= 0.2
 0 <= s_k10_AZN_s41 <= 0.2
 0 <= s_k10_AZN_s42 <= 0.2
 0 <= s_k10_AZN_s43 <= 0.2
 0 <= s_k10_AZN_s44 <= 0.2
 0 <= s_k10_AZN_s45 <= 0.2
 0 <= s_k10_AZN_s46 <= 0.2
 0 <= s_k10_AZN_s47 <= 0.2
 0 <= s_k10_AZN_s48 <= 0.2
 0 <= s_k10_AZN_s49 <= 0.2
 0 <= s_k10_AZN_s50 <= 0.2
 0 <= s_k10_AZN_s51 <= 0.2
 0 <= s_k10_AZN_s52 <= 0.2
 0 <= s_k10_AZN_s53 <= 0.2
 0 <= s_k10_AZN_s54 <= 0.2
 0 <= s_k10_AZN_s55 <= 0.2
 0 <= s_k10_AZN_s56 <= 0.2
 0 <= s_k10_AZN_s57 <= 0.2
 0 <= s_k10_AZN_s58 <= 0.2
 0 <= s_k10_AZN_s59 <= 0.2
 0 <= s_k10_AZN_s60 <= 0.2
 0 <= s_k10_AZN_s61 <= 0.2
 0 <= s_k10_AZN_s62 <= 0.2
 0 <= s_k10_AZN_s63 <= 0.2
 0 <= s_k10_AZN_s64 <= 0.2
 0 <= s_k10_AZN_s65 <= 0.2
 0 <= s_k10_AZN_s66 <= 0.2
 0 <= s_k10_AZN_s67 <= 0.2
 0 <= s_k10_AZN_s68 <= 0.2
 0 <= s_k10_AZN_s69 <= 0.2
 0 <= s_k10_AZN_s70 <= 0.2
 0 <= s_k10_AZN_s71 <= 0.2
 0 <= s_k10_AZN_s72 <= 0.2
 0 <= s_k10_AZN_s73 <= 0.2
 0 <= s_k10_AZN_s74 <= 0.2
 0 <= s_k10_AZN_s75 <= 0.2
 0 <= s_k10_AZN_s76 <= 0.2
 0 <= s_k10_AZN_s77 <= 0.2
 0 <= s_k10_AZN_s78 <= 0.2
 0 <= s_k10_AZN_s79 <= 0.2
 0 <= s_k10_AZN_s80 <= 0.2
 0 <= s_k10_AZN_s81 <= 0.2
 0 <= s_k10_AZN_s82 <= 0.2
 0 <= s_k10_AZN_s83 <= 0.2
 0 <= s_k10_AZN_s84 <= 0.2
 0 <= s_k10_AZN_s85 <= 0.2
 0 <= s_k10_AZN_s86 <= 0.2
 0 <= s_k10_AZN_s87 <= 0.2
 0 <= s_k10_AZN_s88 <= 0.2
 0 <= s_k10_AZN_s89 <= 0.2
 0 <= s_k10_AZN_s90 <= 0.2
 0 <= s_k10_AZN_s91 <= 0.2
 0 <= s_k10_AZN_s92 <= 0.2
 0 <= s_k10_AZN_s93 <= 0.2
 0 <= s_k10_AZN_s94 <= 0.2
 0 <= s_k10_AZN_s95 <= 0.2
 0 <= s_k10_AZN_s96 <= 0.2
 0 <= s_k10_AZN_s97 <= 0.2
 0 <= s_k10_AZN_s98 <= 0.2
 0 <= s_k10_AZN_s99 <= 0.2
 0 <= s_k10_RIO_s0 <= 0.2
 0 <= s_k10_RIO_s1 <= 0.2
 0 <= s_k10_RIO_s2 <= 0.2
 0 <= s_k10_RIO_s3 <= 0.2
 0 <= s_k10_RIO_s4 <= 0.2
 0 <= s_k10_RIO_s5 <= 0.2
 0 <= s_k10_RIO_s6 <= 0.2
 0 <= s_k10_RIO_s7 <= 0.2
 0 <= s_k10_RIO_s8 <= 0.2
 0 <= s_k10_RIO_s9 <= 0.2
 0 <= s_k10_RIO_s00 <= 0.2
 0 <= s_k10_RIO_s01 <= 0.2
 0 <= s_k10_RIO_s02 <= 0.2
 0 <= s_k10_RIO_s03 <= 0.2
 0 <= s_k10_RIO_s04 <= 0.2
 0 <= s_k10_RIO_s05 <= 0.2
 0 <= s_k10_RIO_s06 <= 0.2
 0 <= s_k10_RIO_s07 <= 0.2
 0 <= s_k10_RIO_s08 <= 0.2
 0 <= s_k10_RIO_s09 <= 0.2
 0 <= s_k10_RIO_s10 <= 0.2
 0 <= s_k10_RIO_s11 <= 0.2
 0 <= s_k10_RIO_s12 <= 0.2
 0 <= s_k10_RIO_s13 <= 0.2
 0 <= s_k10_RIO_s14 <= 0.2
 0 <= s_k10_RIO_s15 <= 0.2
 0 <= s_k10_RIO_s16 <= 0.2
 0 <= s_k10_RIO_s17 <= 0.2
 0 <= s_k10_RIO_s18 <= 0.2
 0 <= s_k10_RIO_s19 <= 0.2
 0 <= s_k10_RIO_s20 <= 0.2
 0 <= s_k10_RIO_s21 <= 0.2
 0 <= s_k10_RIO_s22 <= 0.2
 0 <= s_k10_RIO_s23 <= 0.2
 0 <= s_k10_RIO_s24 <= 0.2
 0 <= s_k10_RIO_s25 <= 0.2
 0 <= s_k10_RIO_s26 <= 0.2
 0 <= s_k10_RIO_s27 <= 0.2
 0 <= s_k10_RIO_s28 <= 0.2
 0 <= s_k10_RIO_s29 <= 0.2
 0 <= s_k10_RIO_s30 <= 0.2
 0 <= s_k10_RIO_s31 <= 0.2
 0 <= s_k10_RIO_s32 <= 0.2
 0 <= s_k10_RIO_s33 <= 0.2
 0 <= s_k10_RIO_s34 <= 0.2
 0 <= s_k10_RIO_s35 <= 0.2
 0 <= s_k10_RIO_s36 <= 0.2
 0 <= s_k10_RIO_s37 <= 0.2
 0 <= s_k10_RIO_s38 <= 0.2
 0 <= s_k10_RIO_s39 <= 0.2
 0 <= s_k10_RIO_s40 <= 0.2
 0 <= s_k10_RIO_s41 <= 0.2
 0 <= s_k10_RIO_s42 <= 0.2
 0 <= s_k10_RIO_s43 <= 0.2
 0 <= s_k10_RIO_s44 <= 0.2
 0 <= s_k10_RIO_s45 <= 0.2
 0 <= s_k10_RIO_s46 <= 0.2
 0 <= s_k10_RIO_s47 <= 0.2
 0 <= s_k10_RIO_s48 <= 0.2
 0 <= s_k10_RIO_s49 <= 0.2
 0 <= s_k10_RIO_s50 <= 0.2
 0 <= s_k10_RIO_s51 <= 0.2
 0 <= s_k10_RIO_s52 <= 0.2
 0 <= s_k10_RIO_s53 <= 0.2
 0 <= s_k10_RIO_s54 <= 0.2
 0 <= s_k10_RIO_s55 <= 0.2
 0 <= s_k10_RIO_s56 <= 0.2
 0 <= s_k10_RIO_s57 <= 0.2
 0 <= s_k10_RIO_s58 <= 0.2
 0 <= s_k10_RIO_s59 <= 0.2
 0 <= s_k10_RIO_s60 <= 0.2
 0 <= s_k10_RIO_s61 <= 0.2
 0 <= s_k10_RIO_s62 <= 0.2
 0 <= s_k10_RIO_s63 <= 0.2
 0 <= s_k10_RIO_s64 <= 0.2
 0 <= s_k10_RIO_s65 <= 0.2
 0 <= s_k10_RIO_s66 <= 0.2
 0 <= s_k10_RIO_s67 <= 0.2
 0 <= s_k10_RIO_s68 <= 0.2
 0 <= s_k10_RIO_s69 <= 0.2
 0 <= s_k10_RIO_s70 <= 0.2
 0 <= s_k10_RIO_s71 <= 0.2
 0 <= s_k10_RIO_s72 <= 0.2
 0 <= s_k10_RIO_s73 <= 0.2
 0 <= s_k10_RIO_s74 <= 0.2
 0 <= s_k10_RIO_s75 <= 0.2
 0 <= s_k10_RIO_s76 <= 0.2
 0 <= s_k10_RIO_s77 <= 0.2
 0 <= s_k10_RIO_s78 <= 0.2
 0 <= s_k10_RIO_s79 <= 0.2
 0 <= s_k10_RIO_s80 <= 0.2
 0 <= s_k10_RIO_s81 <= 0.2
 0 <= s_k10_RIO_s82 <= 0.2
 0 <= s_k10_RIO_s83 <= 0.2
 0 <= s_k10_RIO_s84 <= 0.2
 0 <= s_k10_RIO_s85 <= 0.2
 0 <= s_k10_RIO_s86 <= 0.2
 0 <= s_k10_RIO_s87 <= 0.2
 0 <= s_k10_RIO_s88 <= 0.2
 0 <= s_k10_RIO_s89 <= 0.2
 0 <= s_k10_RIO_s90 <= 0.2
 0 <= s_k10_RIO_s91 <= 0.2
 0 <= s_k10_RIO_s92 <= 0.2
 0 <= s_k10_RIO_s93 <= 0.2
 0 <= s_k10_RIO_s94 <= 0.2
 0 <= s_k10_RIO_s95 <= 0.2
 0 <= s_k10_RIO_s96 <= 0.2
 0 <= s_k10_RIO_s97 <= 0.2
 0 <= s_k10_RIO_s98 <= 0.2
 0 <= s_k10_RIO_s99 <= 0.2
 0 <= s_k10_BG_s0 <= 0.2
 0 <= s_k10_BG_s1 <= 0.2
 0 <= s_k10_BG_s2 <= 0.2
 0 <= s_k10_BG_s3 <= 0.2
 0 <= s_k10_BG_s4 <= 0.2
 0 <= s_k10_BG_s5 <= 0.2
 0 <= s_k10_BG_s6 <= 0.2
 0 <= s_k10_BG_s7 <= 0.2
 0 <= s_k10_BG_s8 <= 0.2
 0 <= s_k10_BG_s9 <= 0.2
 0 <= s_k10_BG_s00 <= 0.2
 0 <= s_k10_BG_s01 <= 0.2
 0 <= s_k10_BG_s02 <= 0.2
 0 <= s_k10_BG_s03 <= 0.2
 0 <= s_k10_BG_s04 <= 0.2
 0 <= s_k10_BG_s05 <= 0.2
 0 <= s_k10_BG_s06 <= 0.2
 0 <= s_k10_BG_s07 <= 0.2
 0 <= s_k10_BG_s08 <= 0.2
 0 <= s_k10_BG_s09 <= 0.2
 0 <= s_k10_BG_s10 <= 0.2
 0 <= s_k10_BG_s11 <= 0.2
 0 <= s_k10_BG_s12 <= 0.2
 0 <= s_k10_BG_s13 <= 0.2
 0 <= s_k10_BG_s14 <= 0.2
 0 <= s_k10_BG_s15 <= 0.2
 0 <= s_k10_BG_s16 <= 0.2
 0 <= s_k10_BG_s17 <= 0.2
 0 <= s_k10_BG_s18 <= 0.2
 0 <= s_k10_BG_s19 <= 0.2
 0 <= s_k10_BG_s20 <= 0.2
 0 <= s_k10_BG_s21 <= 0.2
 0 <= s_k10_BG_s22 <= 0.2
 0 <= s_k10_BG_s23 <= 0.2
 0 <= s_k10_BG_s24 <= 0.2
 0 <= s_k10_BG_s25 <= 0.2
 0 <= s_k10_BG_s26 <= 0.2
 0 <= s_k10_BG_s27 <= 0.2
 0 <= s_k10_BG_s28 <= 0.2
 0 <= s_k10_BG_s29 <= 0.2
 0 <= s_k10_BG_s30 <= 0.2
 0 <= s_k10_BG_s31 <= 0.2
 0 <= s_k10_BG_s32 <= 0.2
 0 <= s_k10_BG_s33 <= 0.2
 0 <= s_k10_BG_s34 <= 0.2
 0 <= s_k10_BG_s35 <= 0.2
 0 <= s_k10_BG_s36 <= 0.2
 0 <= s_k10_BG_s37 <= 0.2
 0 <= s_k10_BG_s38 <= 0.2
 0 <= s_k10_BG_s39 <= 0.2
 0 <= s_k10_BG_s40 <= 0.2
 0 <= s_k10_BG_s41 <= 0.2
 0 <= s_k10_BG_s42 <= 0.2
 0 <= s_k10_BG_s43 <= 0.2
 0 <= s_k10_BG_s44 <= 0.2
 0 <= s_k10_BG_s45 <= 0.2
 0 <= s_k10_BG_s46 <= 0.2
 0 <= s_k10_BG_s47 <= 0.2
 0 <= s_k10_BG_s48 <= 0.2
 0 <= s_k10_BG_s49 <= 0.2
 0 <= s_k10_BG_s50 <= 0.2
 0 <= s_k10_BG_s51 <= 0.2
 0 <= s_k10_BG_s52 <= 0.2
 0 <= s_k10_BG_s53 <= 0.2
 0 <= s_k10_BG_s54 <= 0.2
 0 <= s_k10_BG_s55 <= 0.2
 0 <= s_k10_BG_s56 <= 0.2
 0 <= s_k10_BG_s57 <= 0.2
 0 <= s_k10_BG_s58 <= 0.2
 0 <= s_k10_BG_s59 <= 0.2
 0 <= s_k10_BG_s60 <= 0.2
 0 <= s_k10_BG_s61 <= 0.2
 0 <= s_k10_BG_s62 <= 0.2
 0 <= s_k10_BG_s63 <= 0.2
 0 <= s_k10_BG_s64 <= 0.2
 0 <= s_k10_BG_s65 <= 0.2
 0 <= s_k10_BG_s66 <= 0.2
 0 <= s_k10_BG_s67 <= 0.2
 0 <= s_k10_BG_s68 <= 0.2
 0 <= s_k10_BG_s69 <= 0.2
 0 <= s_k10_BG_s70 <= 0.2
 0 <= s_k10_BG_s71 <= 0.2
 0 <= s_k10_BG_s72 <= 0.2
 0 <= s_k10_BG_s73 <= 0.2
 0 <= s_k10_BG_s74 <= 0.2
 0 <= s_k10_BG_s75 <= 0.2
 0 <= s_k10_BG_s76 <= 0.2
 0 <= s_k10_BG_s77 <= 0.2
 0 <= s_k10_BG_s78 <= 0.2
 0 <= s_k10_BG_s79 <= 0.2
 0 <= s_k10_BG_s80 <= 0.2
 0 <= s_k10_BG_s81 <= 0.2
 0 <= s_k10_BG_s82 <= 0.2
 0 <= s_k10_BG_s83 <= 0.2
 0 <= s_k10_BG_s84 <= 0.2
 0 <= s_k10_BG_s85 <= 0.2
 0 <= s_k10_BG_s86 <= 0.2
 0 <= s_k10_BG_s87 <= 0.2
 0 <= s_k10_BG_s88 <= 0.2
 0 <= s_k10_BG_s89 <= 0.2
 0 <= s_k10_BG_s90 <= 0.2
 0 <= s_k10_BG_s91 <= 0.2
 0 <= s_k10_BG_s92 <= 0.2
 0 <= s_k10_BG_s93 <= 0.2
 0 <= s_k10_BG_s94 <= 0.2
 0 <= s_k10_BG_s95 <= 0.2
 0 <= s_k10_BG_s96 <= 0.2
 0 <= s_k10_BG_s97 <= 0.2
 0 <= s_k10_BG_s98 <= 0.2
 0 <= s_k10_BG_s99 <= 0.2
 0 <= s_k10_TSCO_s0 <= 0.2
 0 <= s_k10_TSCO_s1 <= 0.2
 0 <= s_k10_TSCO_s2 <= 0.2
 0 <= s_k10_TSCO_s3 <= 0.2
 0 <= s_k10_TSCO_s4 <= 0.2
 0 <= s_k10_TSCO_s5 <= 0.2
 0 <= s_k10_TSCO_s6 <= 0.2
 0 <= s_k10_TSCO_s7 <= 0.2
 0 <= s_k10_TSCO_s8 <= 0.2
 0 <= s_k10_TSCO_s9 <= 0.2
 0 <= s_k10_TSCO_s00 <= 0.2
 0 <= s_k10_TSCO_s01 <= 0.2
 0 <= s_k10_TSCO_s02 <= 0.2
 0 <= s_k10_TSCO_s03 <= 0.2
 0 <= s_k10_TSCO_s04 <= 0.2
 0 <= s_k10_TSCO_s05 <= 0.2
 0 <= s_k10_TSCO_s06 <= 0.2
 0 <= s_k10_TSCO_s07 <= 0.2
 0 <= s_k10_TSCO_s08 <= 0.2
 0 <= s_k10_TSCO_s09 <= 0.2
 0 <= s_k10_TSCO_s10 <= 0.2
 0 <= s_k10_TSCO_s11 <= 0.2
 0 <= s_k10_TSCO_s12 <= 0.2
 0 <= s_k10_TSCO_s13 <= 0.2
 0 <= s_k10_TSCO_s14 <= 0.2
 0 <= s_k10_TSCO_s15 <= 0.2
 0 <= s_k10_TSCO_s16 <= 0.2
 0 <= s_k10_TSCO_s17 <= 0.2
 0 <= s_k10_TSCO_s18 <= 0.2
 0 <= s_k10_TSCO_s19 <= 0.2
 0 <= s_k10_TSCO_s20 <= 0.2
 0 <= s_k10_TSCO_s21 <= 0.2
 0 <= s_k10_TSCO_s22 <= 0.2
 0 <= s_k10_TSCO_s23 <= 0.2
 0 <= s_k10_TSCO_s24 <= 0.2
 0 <= s_k10_TSCO_s25 <= 0.2
 0 <= s_k10_TSCO_s26 <= 0.2
 0 <= s_k10_TSCO_s27 <= 0.2
 0 <= s_k10_TSCO_s28 <= 0.2
 0 <= s_k10_TSCO_s29 <= 0.2
 0 <= s_k10_TSCO_s30 <= 0.2
 0 <= s_k10_TSCO_s31 <= 0.2
 0 <= s_k10_TSCO_s32 <= 0.2
 0 <= s_k10_TSCO_s33 <= 0.2
 0 <= s_k10_TSCO_s34 <= 0.2
 0 <= s_k10_TSCO_s35 <= 0.2
 0 <= s_k10_TSCO_s36 <= 0.2
 0 <= s_k10_TSCO_s37 <= 0.2
 0 <= s_k10_TSCO_s38 <= 0.2
 0 <= s_k10_TSCO_s39 <= 0.2
 0 <= s_k10_TSCO_s40 <= 0.2
 0 <= s_k10_TSCO_s41 <= 0.2
 0 <= s_k10_TSCO_s42 <= 0.2
 0 <= s_k10_TSCO_s43 <= 0.2
 0 <= s_k10_TSCO_s44 <= 0.2
 0 <= s_k10_TSCO_s45 <= 0.2
 0 <= s_k10_TSCO_s46 <= 0.2
 0 <= s_k10_TSCO_s47 <= 0.2
 0 <= s_k10_TSCO_s48 <= 0.2
 0 <= s_k10_TSCO_s49 <= 0.2
 0 <= s_k10_TSCO_s50 <= 0.2
 0 <= s_k10_TSCO_s51 <= 0.2
 0 <= s_k10_TSCO_s52 <= 0.2
 0 <= s_k10_TSCO_s53 <= 0.2
 0 <= s_k10_TSCO_s54 <= 0.2
 0 <= s_k10_TSCO_s55 <= 0.2
 0 <= s_k10_TSCO_s56 <= 0.2
 0 <= s_k10_TSCO_s57 <= 0.2
 0 <= s_k10_TSCO_s58 <= 0.2
 0 <= s_k10_TSCO_s59 <= 0.2
 0 <= s_k10_TSCO_s60 <= 0.2
 0 <= s_k10_TSCO_s61 <= 0.2
 0 <= s_k10_TSCO_s62 <= 0.2
 0 <= s_k10_TSCO_s63 <= 0.2
 0 <= s_k10_TSCO_s64 <= 0.2
 0 <= s_k10_TSCO_s65 <= 0.2
 0 <= s_k10_TSCO_s66 <= 0.2
 0 <= s_k10_TSCO_s67 <= 0.2
 0 <= s_k10_TSCO_s68 <= 0.2
 0 <= s_k10_TSCO_s69 <= 0.2
 0 <= s_k10_TSCO_s70 <= 0.2
 0 <= s_k10_TSCO_s71 <= 0.2
 0 <= s_k10_TSCO_s72 <= 0.2
 0 <= s_k10_TSCO_s73 <= 0.2
 0 <= s_k10_TSCO_s74 <= 0.2
 0 <= s_k10_TSCO_s75 <= 0.2
 0 <= s_k10_TSCO_s76 <= 0.2
 0 <= s_k10_TSCO_s77 <= 0.2
 0 <= s_k10_TSCO_s78 <= 0.2
 0 <= s_k10_TSCO_s79 <= 0.2
 0 <= s_k10_TSCO_s80 <= 0.2
 0 <= s_k10_TSCO_s81 <= 0.2
 0 <= s_k10_TSCO_s82 <= 0.2
 0 <= s_k10_TSCO_s83 <= 0.2
 0 <= s_k10_TSCO_s84 <= 0.2
 0 <= s_k10_TSCO_s85 <= 0.2
 0 <= s_k10_TSCO_s86 <= 0.2
 0 <= s_k10_TSCO_s87 <= 0.2
 0 <= s_k10_TSCO_s88 <= 0.2
 0 <= s_k10_TSCO_s89 <= 0.2
 0 <= s_k10_TSCO_s90 <= 0.2
 0 <= s_k10_TSCO_s91 <= 0.2
 0 <= s_k10_TSCO_s92 <= 0.2
 0 <= s_k10_TSCO_s93 <= 0.2
 0 <= s_k10_TSCO_s94 <= 0.2
 0 <= s_k10_TSCO_s95 <= 0.2
 0 <= s_k10_TSCO_s96 <= 0.2
 0 <= s_k10_TSCO_s97 <= 0.2
 0 <= s_k10_TSCO_s98 <= 0.2
 0 <= s_k10_TSCO_s99 <= 0.2
 0 <= s_k10_AAL_s0 <= 0.2
 0 <= s_k10_AAL_s1 <= 0.2
 0 <= s_k10_AAL_s2 <= 0.2
 0 <= s_k10_AAL_s3 <= 0.2
 0 <= s_k10_AAL_s4 <= 0.2
 0 <= s_k10_AAL_s5 <= 0.2
 0 <= s_k10_AAL_s6 <= 0.2
 0 <= s_k10_AAL_s7 <= 0.2
 0 <= s_k10_AAL_s8 <= 0.2
 0 <= s_k10_AAL_s9 <= 0.2
 0 <= s_k10_AAL_s00 <= 0.2
 0 <= s_k10_AAL_s01 <= 0.2
 0 <= s_k10_AAL_s02 <= 0.2
 0 <= s_k10_AAL_s03 <= 0.2
 0 <= s_k10_AAL_s04 <= 0.2
 0 <= s_k10_AAL_s05 <= 0.2
 0 <= s_k10_AAL_s06 <= 0.2
 0 <= s_k10_AAL_s07 <= 0.2
 0 <= s_k10_AAL_s08 <= 0.2
 0 <= s_k10_AAL_s09 <= 0.2
 0 <= s_k10_AAL_s10 <= 0.2
 0 <= s_k10_AAL_s11 <= 0.2
 0 <= s_k10_AAL_s12 <= 0.2
 0 <= s_k10_AAL_s13 <= 0.2
 0 <= s_k10_AAL_s14 <= 0.2
 0 <= s_k10_AAL_s15 <= 0.2
 0 <= s_k10_AAL_s16 <= 0.2
 0 <= s_k10_AAL_s17 <= 0.2
 0 <= s_k10_AAL_s18 <= 0.2
 0 <= s_k10_AAL_s19 <= 0.2
 0 <= s_k10_AAL_s20 <= 0.2
 0 <= s_k10_AAL_s21 <= 0.2
 0 <= s_k10_AAL_s22 <= 0.2
 0 <= s_k10_AAL_s23 <= 0.2
 0 <= s_k10_AAL_s24 <= 0.2
 0 <= s_k10_AAL_s25 <= 0.2
 0 <= s_k10_AAL_s26 <= 0.2
 0 <= s_k10_AAL_s27 <= 0.2
 0 <= s_k10_AAL_s28 <= 0.2
 0 <= s_k10_AAL_s29 <= 0.2
 0 <= s_k10_AAL_s30 <= 0.2
 0 <= s_k10_AAL_s31 <= 0.2
 0 <= s_k10_AAL_s32 <= 0.2
 0 <= s_k10_AAL_s33 <= 0.2
 0 <= s_k10_AAL_s34 <= 0.2
 0 <= s_k10_AAL_s35 <= 0.2
 0 <= s_k10_AAL_s36 <= 0.2
 0 <= s_k10_AAL_s37 <= 0.2
 0 <= s_k10_AAL_s38 <= 0.2
 0 <= s_k10_AAL_s39 <= 0.2
 0 <= s_k10_AAL_s40 <= 0.2
 0 <= s_k10_AAL_s41 <= 0.2
 0 <= s_k10_AAL_s42 <= 0.2
 0 <= s_k10_AAL_s43 <= 0.2
 0 <= s_k10_AAL_s44 <= 0.2
 0 <= s_k10_AAL_s45 <= 0.2
 0 <= s_k10_AAL_s46 <= 0.2
 0 <= s_k10_AAL_s47 <= 0.2
 0 <= s_k10_AAL_s48 <= 0.2
 0 <= s_k10_AAL_s49 <= 0.2
 0 <= s_k10_AAL_s50 <= 0.2
 0 <= s_k10_AAL_s51 <= 0.2
 0 <= s_k10_AAL_s52 <= 0.2
 0 <= s_k10_AAL_s53 <= 0.2
 0 <= s_k10_AAL_s54 <= 0.2
 0 <= s_k10_AAL_s55 <= 0.2
 0 <= s_k10_AAL_s56 <= 0.2
 0 <= s_k10_AAL_s57 <= 0.2
 0 <= s_k10_AAL_s58 <= 0.2
 0 <= s_k10_AAL_s59 <= 0.2
 0 <= s_k10_AAL_s60 <= 0.2
 0 <= s_k10_AAL_s61 <= 0.2
 0 <= s_k10_AAL_s62 <= 0.2
 0 <= s_k10_AAL_s63 <= 0.2
 0 <= s_k10_AAL_s64 <= 0.2
 0 <= s_k10_AAL_s65 <= 0.2
 0 <= s_k10_AAL_s66 <= 0.2
 0 <= s_k10_AAL_s67 <= 0.2
 0 <= s_k10_AAL_s68 <= 0.2
 0 <= s_k10_AAL_s69 <= 0.2
 0 <= s_k10_AAL_s70 <= 0.2
 0 <= s_k10_AAL_s71 <= 0.2
 0 <= s_k10_AAL_s72 <= 0.2
 0 <= s_k10_AAL_s73 <= 0.2
 0 <= s_k10_AAL_s74 <= 0.2
 0 <= s_k10_AAL_s75 <= 0.2
 0 <= s_k10_AAL_s76 <= 0.2
 0 <= s_k10_AAL_s77 <= 0.2
 0 <= s_k10_AAL_s78 <= 0.2
 0 <= s_k10_AAL_s79 <= 0.2
 0 <= s_k10_AAL_s80 <= 0.2
 0 <= s_k10_AAL_s81 <= 0.2
 0 <= s_k10_AAL_s82 <= 0.2
 0 <= s_k10_AAL_s83 <= 0.2
 0 <= s_k10_AAL_s84 <= 0.2
 0 <= s_k10_AAL_s85 <= 0.2
 0 <= s_k10_AAL_s86 <= 0.2
 0 <= s_k10_AAL_s87 <= 0.2
 0 <= s_k10_AAL_s88 <= 0.2
 0 <= s_k10_AAL_s89 <= 0.2
 0 <= s_k10_AAL_s90 <= 0.2
 0 <= s_k10_AAL_s91 <= 0.2
 0 <= s_k10_AAL_s92 <= 0.2
 0 <= s_k10_AAL_s93 <= 0.2
 0 <= s_k10_AAL_s94 <= 0.2
 0 <= s_k10_AAL_s95 <= 0.2
 0 <= s_k10_AAL_s96 <= 0.2
 0 <= s_k10_AAL_s97 <= 0.2
 0 <= s_k10_AAL_s98 <= 0.2
 0 <= s_k10_AAL_s99 <= 0.2
 0 <= s_k10_PRU_s0 <= 0.2
 0 <= s_k10_PRU_s1 <= 0.2
 0 <= s_k10_PRU_s2 <= 0.2
 0 <= s_k10_PRU_s3 <= 0.2
 0 <= s_k10_PRU_s4 <= 0.2
 0 <= s_k10_PRU_s5 <= 0.2
 0 <= s_k10_PRU_s6 <= 0.2
 0 <= s_k10_PRU_s7 <= 0.2
 0 <= s_k10_PRU_s8 <= 0.2
 0 <= s_k10_PRU_s9 <= 0.2
 0 <= s_k10_PRU_s00 <= 0.2
 0 <= s_k10_PRU_s01 <= 0.2
 0 <= s_k10_PRU_s02 <= 0.2
 0 <= s_k10_PRU_s03 <= 0.2
 0 <= s_k10_PRU_s04 <= 0.2
 0 <= s_k10_PRU_s05 <= 0.2
 0 <= s_k10_PRU_s06 <= 0.2
 0 <= s_k10_PRU_s07 <= 0.2
 0 <= s_k10_PRU_s08 <= 0.2
 0 <= s_k10_PRU_s09 <= 0.2
 0 <= s_k10_PRU_s10 <= 0.2
 0 <= s_k10_PRU_s11 <= 0.2
 0 <= s_k10_PRU_s12 <= 0.2
 0 <= s_k10_PRU_s13 <= 0.2
 0 <= s_k10_PRU_s14 <= 0.2
 0 <= s_k10_PRU_s15 <= 0.2
 0 <= s_k10_PRU_s16 <= 0.2
 0 <= s_k10_PRU_s17 <= 0.2
 0 <= s_k10_PRU_s18 <= 0.2
 0 <= s_k10_PRU_s19 <= 0.2
 0 <= s_k10_PRU_s20 <= 0.2
 0 <= s_k10_PRU_s21 <= 0.2
 0 <= s_k10_PRU_s22 <= 0.2
 0 <= s_k10_PRU_s23 <= 0.2
 0 <= s_k10_PRU_s24 <= 0.2
 0 <= s_k10_PRU_s25 <= 0.2
 0 <= s_k10_PRU_s26 <= 0.2
 0 <= s_k10_PRU_s27 <= 0.2
 0 <= s_k10_PRU_s28 <= 0.2
 0 <= s_k10_PRU_s29 <= 0.2
 0 <= s_k10_PRU_s30 <= 0.2
 0 <= s_k10_PRU_s31 <= 0.2
 0 <= s_k10_PRU_s32 <= 0.2
 0 <= s_k10_PRU_s33 <= 0.2
 0 <= s_k10_PRU_s34 <= 0.2
 0 <= s_k10_PRU_s35 <= 0.2
 0 <= s_k10_PRU_s36 <= 0.2
 0 <= s_k10_PRU_s37 <= 0.2
 0 <= s_k10_PRU_s38 <= 0.2
 0 <= s_k10_PRU_s39 <= 0.2
 0 <= s_k10_PRU_s40 <= 0.2
 0 <= s_k10_PRU_s41 <= 0.2
 0 <= s_k10_PRU_s42 <= 0.2
 0 <= s_k10_PRU_s43 <= 0.2
 0 <= s_k10_PRU_s44 <= 0.2
 0 <= s_k10_PRU_s45 <= 0.2
 0 <= s_k10_PRU_s46 <= 0.2
 0 <= s_k10_PRU_s47 <= 0.2
 0 <= s_k10_PRU_s48 <= 0.2
 0 <= s_k10_PRU_s49 <= 0.2
 0 <= s_k10_PRU_s50 <= 0.2
 0 <= s_k10_PRU_s51 <= 0.2
 0 <= s_k10_PRU_s52 <= 0.2
 0 <= s_k10_PRU_s53 <= 0.2
 0 <= s_k10_PRU_s54 <= 0.2
 0 <= s_k10_PRU_s55 <= 0.2
 0 <= s_k10_PRU_s56 <= 0.2
 0 <= s_k10_PRU_s57 <= 0.2
 0 <= s_k10_PRU_s58 <= 0.2
 0 <= s_k10_PRU_s59 <= 0.2
 0 <= s_k10_PRU_s60 <= 0.2
 0 <= s_k10_PRU_s61 <= 0.2
 0 <= s_k10_PRU_s62 <= 0.2
 0 <= s_k10_PRU_s63 <= 0.2
 0 <= s_k10_PRU_s64 <= 0.2
 0 <= s_k10_PRU_s65 <= 0.2
 0 <= s_k10_PRU_s66 <= 0.2
 0 <= s_k10_PRU_s67 <= 0.2
 0 <= s_k10_PRU_s68 <= 0.2
 0 <= s_k10_PRU_s69 <= 0.2
 0 <= s_k10_PRU_s70 <= 0.2
 0 <= s_k10_PRU_s71 <= 0.2
 0 <= s_k10_PRU_s72 <= 0.2
 0 <= s_k10_PRU_s73 <= 0.2
 0 <= s_k10_PRU_s74 <= 0.2
 0 <= s_k10_PRU_s75 <= 0.2
 0 <= s_k10_PRU_s76 <= 0.2
 0 <= s_k10_PRU_s77 <= 0.2
 0 <= s_k10_PRU_s78 <= 0.2
 0 <= s_k10_PRU_s79 <= 0.2
 0 <= s_k10_PRU_s80 <= 0.2
 0 <= s_k10_PRU_s81 <= 0.2
 0 <= s_k10_PRU_s82 <= 0.2
 0 <= s_k10_PRU_s83 <= 0.2
 0 <= s_k10_PRU_s84 <= 0.2
 0 <= s_k10_PRU_s85 <= 0.2
 0 <= s_k10_PRU_s86 <= 0.2
 0 <= s_k10_PRU_s87 <= 0.2
 0 <= s_k10_PRU_s88 <= 0.2
 0 <= s_k10_PRU_s89 <= 0.2
 0 <= s_k10_PRU_s90 <= 0.2
 0 <= s_k10_PRU_s91 <= 0.2
 0 <= s_k10_PRU_s92 <= 0.2
 0 <= s_k10_PRU_s93 <= 0.2
 0 <= s_k10_PRU_s94 <= 0.2
 0 <= s_k10_PRU_s95 <= 0.2
 0 <= s_k10_PRU_s96 <= 0.2
 0 <= s_k10_PRU_s97 <= 0.2
 0 <= s_k10_PRU_s98 <= 0.2
 0 <= s_k10_PRU_s99 <= 0.2
End
