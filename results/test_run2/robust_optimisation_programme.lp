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
 w_k10_HSBC_s0_return_constraint: 1.09402 w_0_HSBC - w_k10_HSBC_s0
                                 + 0.99 b_k10_HSBC_s0 - 1.01 s_k10_HSBC_s0  = 0
 w_k10_HSBC_s1_return_constraint: 0.956206 w_0_HSBC - w_k10_HSBC_s1
                                 + 0.99 b_k10_HSBC_s1 - 1.01 s_k10_HSBC_s1  = 0
 w_k10_HSBC_s00_return_constraint: 1.00728 w_k10_HSBC_s0 - w_k10_HSBC_s00
                                 + 0.99 b_k10_HSBC_s00 - 1.01 s_k10_HSBC_s00
                                  = 0
 w_k10_HSBC_s01_return_constraint: 1.0677 w_k10_HSBC_s0 - w_k10_HSBC_s01
                                 + 0.99 b_k10_HSBC_s01 - 1.01 s_k10_HSBC_s01
                                  = 0
 w_k10_HSBC_s10_return_constraint: 0.940711 w_k10_HSBC_s1 - w_k10_HSBC_s10
                                 + 0.99 b_k10_HSBC_s10 - 1.01 s_k10_HSBC_s10
                                  = 0
 w_k10_HSBC_s11_return_constraint: 1.15386 w_k10_HSBC_s1 - w_k10_HSBC_s11
                                 + 0.99 b_k10_HSBC_s11 - 1.01 s_k10_HSBC_s11
                                  = 0
 w_k10_HSBC_s000_return_constraint: 1.2345 w_k10_HSBC_s00 - w_k10_HSBC_s000
                                 + 0.99 b_k10_HSBC_s000 - 1.01 s_k10_HSBC_s000
                                  = 0
 w_k10_HSBC_s001_return_constraint: 0.849562 w_k10_HSBC_s00 - w_k10_HSBC_s001
                                 + 0.99 b_k10_HSBC_s001 - 1.01 s_k10_HSBC_s001
                                  = 0
 w_k10_HSBC_s002_return_constraint: 0.853149 w_k10_HSBC_s00 - w_k10_HSBC_s002
                                 + 0.99 b_k10_HSBC_s002 - 1.01 s_k10_HSBC_s002
                                  = 0
 w_k10_HSBC_s003_return_constraint: 1.03189 w_k10_HSBC_s00 - w_k10_HSBC_s003
                                 + 0.99 b_k10_HSBC_s003 - 1.01 s_k10_HSBC_s003
                                  = 0
 w_k10_HSBC_s004_return_constraint: 1.04017 w_k10_HSBC_s00 - w_k10_HSBC_s004
                                 + 0.99 b_k10_HSBC_s004 - 1.01 s_k10_HSBC_s004
                                  = 0
 w_k10_HSBC_s005_return_constraint: 0.897439 w_k10_HSBC_s00 - w_k10_HSBC_s005
                                 + 0.99 b_k10_HSBC_s005 - 1.01 s_k10_HSBC_s005
                                  = 0
 w_k10_HSBC_s006_return_constraint: 0.820026 w_k10_HSBC_s00 - w_k10_HSBC_s006
                                 + 0.99 b_k10_HSBC_s006 - 1.01 s_k10_HSBC_s006
                                  = 0
 w_k10_HSBC_s007_return_constraint: 1.02773 w_k10_HSBC_s00 - w_k10_HSBC_s007
                                 + 0.99 b_k10_HSBC_s007 - 1.01 s_k10_HSBC_s007
                                  = 0
 w_k10_HSBC_s010_return_constraint: 1.14512 w_k10_HSBC_s01 - w_k10_HSBC_s010
                                 + 0.99 b_k10_HSBC_s010 - 1.01 s_k10_HSBC_s010
                                  = 0
 w_k10_HSBC_s011_return_constraint: 1.36315 w_k10_HSBC_s01 - w_k10_HSBC_s011
                                 + 0.99 b_k10_HSBC_s011 - 1.01 s_k10_HSBC_s011
                                  = 0
 w_k10_HSBC_s012_return_constraint: 0.953644 w_k10_HSBC_s01 - w_k10_HSBC_s012
                                 + 0.99 b_k10_HSBC_s012 - 1.01 s_k10_HSBC_s012
                                  = 0
 w_k10_HSBC_s013_return_constraint: 0.700967 w_k10_HSBC_s01 - w_k10_HSBC_s013
                                 + 0.99 b_k10_HSBC_s013 - 1.01 s_k10_HSBC_s013
                                  = 0
 w_k10_HSBC_s014_return_constraint: 0.942323 w_k10_HSBC_s01 - w_k10_HSBC_s014
                                 + 0.99 b_k10_HSBC_s014 - 1.01 s_k10_HSBC_s014
                                  = 0
 w_k10_HSBC_s015_return_constraint: 1.02204 w_k10_HSBC_s01 - w_k10_HSBC_s015
                                 + 0.99 b_k10_HSBC_s015 - 1.01 s_k10_HSBC_s015
                                  = 0
 w_k10_HSBC_s016_return_constraint: 0.848799 w_k10_HSBC_s01 - w_k10_HSBC_s016
                                 + 0.99 b_k10_HSBC_s016 - 1.01 s_k10_HSBC_s016
                                  = 0
 w_k10_HSBC_s017_return_constraint: 1.11361 w_k10_HSBC_s01 - w_k10_HSBC_s017
                                 + 0.99 b_k10_HSBC_s017 - 1.01 s_k10_HSBC_s017
                                  = 0
 w_k10_HSBC_s100_return_constraint: 1.10073 w_k10_HSBC_s10 - w_k10_HSBC_s100
                                 + 0.99 b_k10_HSBC_s100 - 1.01 s_k10_HSBC_s100
                                  = 0
 w_k10_HSBC_s101_return_constraint: 0.903362 w_k10_HSBC_s10 - w_k10_HSBC_s101
                                 + 0.99 b_k10_HSBC_s101 - 1.01 s_k10_HSBC_s101
                                  = 0
 w_k10_HSBC_s102_return_constraint: 0.908167 w_k10_HSBC_s10 - w_k10_HSBC_s102
                                 + 0.99 b_k10_HSBC_s102 - 1.01 s_k10_HSBC_s102
                                  = 0
 w_k10_HSBC_s103_return_constraint: 0.866316 w_k10_HSBC_s10 - w_k10_HSBC_s103
                                 + 0.99 b_k10_HSBC_s103 - 1.01 s_k10_HSBC_s103
                                  = 0
 w_k10_HSBC_s104_return_constraint: 0.968488 w_k10_HSBC_s10 - w_k10_HSBC_s104
                                 + 0.99 b_k10_HSBC_s104 - 1.01 s_k10_HSBC_s104
                                  = 0
 w_k10_HSBC_s105_return_constraint: 1.35528 w_k10_HSBC_s10 - w_k10_HSBC_s105
                                 + 0.99 b_k10_HSBC_s105 - 1.01 s_k10_HSBC_s105
                                  = 0
 w_k10_HSBC_s106_return_constraint: 1.18851 w_k10_HSBC_s10 - w_k10_HSBC_s106
                                 + 0.99 b_k10_HSBC_s106 - 1.01 s_k10_HSBC_s106
                                  = 0
 w_k10_HSBC_s107_return_constraint: 1.08058 w_k10_HSBC_s10 - w_k10_HSBC_s107
                                 + 0.99 b_k10_HSBC_s107 - 1.01 s_k10_HSBC_s107
                                  = 0
 w_k10_HSBC_s110_return_constraint: 0.958682 w_k10_HSBC_s11 - w_k10_HSBC_s110
                                 + 0.99 b_k10_HSBC_s110 - 1.01 s_k10_HSBC_s110
                                  = 0
 w_k10_HSBC_s111_return_constraint: 0.867588 w_k10_HSBC_s11 - w_k10_HSBC_s111
                                 + 0.99 b_k10_HSBC_s111 - 1.01 s_k10_HSBC_s111
                                  = 0
 w_k10_HSBC_s112_return_constraint: 1.01168 w_k10_HSBC_s11 - w_k10_HSBC_s112
                                 + 0.99 b_k10_HSBC_s112 - 1.01 s_k10_HSBC_s112
                                  = 0
 w_k10_HSBC_s113_return_constraint: 0.870336 w_k10_HSBC_s11 - w_k10_HSBC_s113
                                 + 0.99 b_k10_HSBC_s113 - 1.01 s_k10_HSBC_s113
                                  = 0
 w_k10_HSBC_s114_return_constraint: 1.20646 w_k10_HSBC_s11 - w_k10_HSBC_s114
                                 + 0.99 b_k10_HSBC_s114 - 1.01 s_k10_HSBC_s114
                                  = 0
 w_k10_HSBC_s115_return_constraint: 0.887678 w_k10_HSBC_s11 - w_k10_HSBC_s115
                                 + 0.99 b_k10_HSBC_s115 - 1.01 s_k10_HSBC_s115
                                  = 0
 w_k10_HSBC_s116_return_constraint: 0.995078 w_k10_HSBC_s11 - w_k10_HSBC_s116
                                 + 0.99 b_k10_HSBC_s116 - 1.01 s_k10_HSBC_s116
                                  = 0
 w_k10_HSBC_s117_return_constraint: 0.933172 w_k10_HSBC_s11 - w_k10_HSBC_s117
                                 + 0.99 b_k10_HSBC_s117 - 1.01 s_k10_HSBC_s117
                                  = 0
 w_k10_VOD_s0_return_constraint: 0.993369 w_0_VOD - w_k10_VOD_s0
                                 + 0.99 b_k10_VOD_s0 - 1.01 s_k10_VOD_s0  = 0
 w_k10_VOD_s1_return_constraint: 0.965805 w_0_VOD - w_k10_VOD_s1
                                 + 0.99 b_k10_VOD_s1 - 1.01 s_k10_VOD_s1  = 0
 w_k10_VOD_s00_return_constraint: 1.17539 w_k10_VOD_s0 - w_k10_VOD_s00
                                 + 0.99 b_k10_VOD_s00 - 1.01 s_k10_VOD_s00  = 0
 w_k10_VOD_s01_return_constraint: 0.925423 w_k10_VOD_s0 - w_k10_VOD_s01
                                 + 0.99 b_k10_VOD_s01 - 1.01 s_k10_VOD_s01  = 0
 w_k10_VOD_s10_return_constraint: 1.00942 w_k10_VOD_s1 - w_k10_VOD_s10
                                 + 0.99 b_k10_VOD_s10 - 1.01 s_k10_VOD_s10  = 0
 w_k10_VOD_s11_return_constraint: 1.09612 w_k10_VOD_s1 - w_k10_VOD_s11
                                 + 0.99 b_k10_VOD_s11 - 1.01 s_k10_VOD_s11  = 0
 w_k10_VOD_s000_return_constraint: 0.91235 w_k10_VOD_s00 - w_k10_VOD_s000
                                 + 0.99 b_k10_VOD_s000 - 1.01 s_k10_VOD_s000
                                  = 0
 w_k10_VOD_s001_return_constraint: 0.787741 w_k10_VOD_s00 - w_k10_VOD_s001
                                 + 0.99 b_k10_VOD_s001 - 1.01 s_k10_VOD_s001
                                  = 0
 w_k10_VOD_s002_return_constraint: 0.795831 w_k10_VOD_s00 - w_k10_VOD_s002
                                 + 0.99 b_k10_VOD_s002 - 1.01 s_k10_VOD_s002
                                  = 0
 w_k10_VOD_s003_return_constraint: 0.973185 w_k10_VOD_s00 - w_k10_VOD_s003
                                 + 0.99 b_k10_VOD_s003 - 1.01 s_k10_VOD_s003
                                  = 0
 w_k10_VOD_s004_return_constraint: 0.924857 w_k10_VOD_s00 - w_k10_VOD_s004
                                 + 0.99 b_k10_VOD_s004 - 1.01 s_k10_VOD_s004
                                  = 0
 w_k10_VOD_s005_return_constraint: 0.967396 w_k10_VOD_s00 - w_k10_VOD_s005
                                 + 0.99 b_k10_VOD_s005 - 1.01 s_k10_VOD_s005
                                  = 0
 w_k10_VOD_s006_return_constraint: 0.852754 w_k10_VOD_s00 - w_k10_VOD_s006
                                 + 0.99 b_k10_VOD_s006 - 1.01 s_k10_VOD_s006
                                  = 0
 w_k10_VOD_s007_return_constraint: 0.98906 w_k10_VOD_s00 - w_k10_VOD_s007
                                 + 0.99 b_k10_VOD_s007 - 1.01 s_k10_VOD_s007
                                  = 0
 w_k10_VOD_s010_return_constraint: 1.14914 w_k10_VOD_s01 - w_k10_VOD_s010
                                 + 0.99 b_k10_VOD_s010 - 1.01 s_k10_VOD_s010
                                  = 0
 w_k10_VOD_s011_return_constraint: 1.26119 w_k10_VOD_s01 - w_k10_VOD_s011
                                 + 0.99 b_k10_VOD_s011 - 1.01 s_k10_VOD_s011
                                  = 0
 w_k10_VOD_s012_return_constraint: 1.12813 w_k10_VOD_s01 - w_k10_VOD_s012
                                 + 0.99 b_k10_VOD_s012 - 1.01 s_k10_VOD_s012
                                  = 0
 w_k10_VOD_s013_return_constraint: 1.12474 w_k10_VOD_s01 - w_k10_VOD_s013
                                 + 0.99 b_k10_VOD_s013 - 1.01 s_k10_VOD_s013
                                  = 0
 w_k10_VOD_s014_return_constraint: 1.15043 w_k10_VOD_s01 - w_k10_VOD_s014
                                 + 0.99 b_k10_VOD_s014 - 1.01 s_k10_VOD_s014
                                  = 0
 w_k10_VOD_s015_return_constraint: 1.08324 w_k10_VOD_s01 - w_k10_VOD_s015
                                 + 0.99 b_k10_VOD_s015 - 1.01 s_k10_VOD_s015
                                  = 0
 w_k10_VOD_s016_return_constraint: 0.94308 w_k10_VOD_s01 - w_k10_VOD_s016
                                 + 0.99 b_k10_VOD_s016 - 1.01 s_k10_VOD_s016
                                  = 0
 w_k10_VOD_s017_return_constraint: 1.41584 w_k10_VOD_s01 - w_k10_VOD_s017
                                 + 0.99 b_k10_VOD_s017 - 1.01 s_k10_VOD_s017
                                  = 0
 w_k10_VOD_s100_return_constraint: 1.13036 w_k10_VOD_s10 - w_k10_VOD_s100
                                 + 0.99 b_k10_VOD_s100 - 1.01 s_k10_VOD_s100
                                  = 0
 w_k10_VOD_s101_return_constraint: 0.882384 w_k10_VOD_s10 - w_k10_VOD_s101
                                 + 0.99 b_k10_VOD_s101 - 1.01 s_k10_VOD_s101
                                  = 0
 w_k10_VOD_s102_return_constraint: 0.906956 w_k10_VOD_s10 - w_k10_VOD_s102
                                 + 0.99 b_k10_VOD_s102 - 1.01 s_k10_VOD_s102
                                  = 0
 w_k10_VOD_s103_return_constraint: 0.906279 w_k10_VOD_s10 - w_k10_VOD_s103
                                 + 0.99 b_k10_VOD_s103 - 1.01 s_k10_VOD_s103
                                  = 0
 w_k10_VOD_s104_return_constraint: 0.866022 w_k10_VOD_s10 - w_k10_VOD_s104
                                 + 0.99 b_k10_VOD_s104 - 1.01 s_k10_VOD_s104
                                  = 0
 w_k10_VOD_s105_return_constraint: 1.13374 w_k10_VOD_s10 - w_k10_VOD_s105
                                 + 0.99 b_k10_VOD_s105 - 1.01 s_k10_VOD_s105
                                  = 0
 w_k10_VOD_s106_return_constraint: 0.997209 w_k10_VOD_s10 - w_k10_VOD_s106
                                 + 0.99 b_k10_VOD_s106 - 1.01 s_k10_VOD_s106
                                  = 0
 w_k10_VOD_s107_return_constraint: 1.06739 w_k10_VOD_s10 - w_k10_VOD_s107
                                 + 0.99 b_k10_VOD_s107 - 1.01 s_k10_VOD_s107
                                  = 0
 w_k10_VOD_s110_return_constraint: 1.21018 w_k10_VOD_s11 - w_k10_VOD_s110
                                 + 0.99 b_k10_VOD_s110 - 1.01 s_k10_VOD_s110
                                  = 0
 w_k10_VOD_s111_return_constraint: 1.04444 w_k10_VOD_s11 - w_k10_VOD_s111
                                 + 0.99 b_k10_VOD_s111 - 1.01 s_k10_VOD_s111
                                  = 0
 w_k10_VOD_s112_return_constraint: 0.919314 w_k10_VOD_s11 - w_k10_VOD_s112
                                 + 0.99 b_k10_VOD_s112 - 1.01 s_k10_VOD_s112
                                  = 0
 w_k10_VOD_s113_return_constraint: 1.00757 w_k10_VOD_s11 - w_k10_VOD_s113
                                 + 0.99 b_k10_VOD_s113 - 1.01 s_k10_VOD_s113
                                  = 0
 w_k10_VOD_s114_return_constraint: 1.03989 w_k10_VOD_s11 - w_k10_VOD_s114
                                 + 0.99 b_k10_VOD_s114 - 1.01 s_k10_VOD_s114
                                  = 0
 w_k10_VOD_s115_return_constraint: 1.10892 w_k10_VOD_s11 - w_k10_VOD_s115
                                 + 0.99 b_k10_VOD_s115 - 1.01 s_k10_VOD_s115
                                  = 0
 w_k10_VOD_s116_return_constraint: 1.08848 w_k10_VOD_s11 - w_k10_VOD_s116
                                 + 0.99 b_k10_VOD_s116 - 1.01 s_k10_VOD_s116
                                  = 0
 w_k10_VOD_s117_return_constraint: 1.05602 w_k10_VOD_s11 - w_k10_VOD_s117
                                 + 0.99 b_k10_VOD_s117 - 1.01 s_k10_VOD_s117
                                  = 0
 w_k10_BP_s0_return_constraint:  1.15535 w_0_BP - w_k10_BP_s0
                                 + 0.99 b_k10_BP_s0 - 1.01 s_k10_BP_s0  = 0
 w_k10_BP_s1_return_constraint:  0.927829 w_0_BP - w_k10_BP_s1
                                 + 0.99 b_k10_BP_s1 - 1.01 s_k10_BP_s1  = 0
 w_k10_BP_s00_return_constraint: 1.01612 w_k10_BP_s0 - w_k10_BP_s00
                                 + 0.99 b_k10_BP_s00 - 1.01 s_k10_BP_s00  = 0
 w_k10_BP_s01_return_constraint: 0.983539 w_k10_BP_s0 - w_k10_BP_s01
                                 + 0.99 b_k10_BP_s01 - 1.01 s_k10_BP_s01  = 0
 w_k10_BP_s10_return_constraint: 0.929642 w_k10_BP_s1 - w_k10_BP_s10
                                 + 0.99 b_k10_BP_s10 - 1.01 s_k10_BP_s10  = 0
 w_k10_BP_s11_return_constraint: 1.30429 w_k10_BP_s1 - w_k10_BP_s11
                                 + 0.99 b_k10_BP_s11 - 1.01 s_k10_BP_s11  = 0
 w_k10_BP_s000_return_constraint: 1.64864 w_k10_BP_s00 - w_k10_BP_s000
                                 + 0.99 b_k10_BP_s000 - 1.01 s_k10_BP_s000  = 0
 w_k10_BP_s001_return_constraint: 0.726163 w_k10_BP_s00 - w_k10_BP_s001
                                 + 0.99 b_k10_BP_s001 - 1.01 s_k10_BP_s001  = 0
 w_k10_BP_s002_return_constraint: 0.902915 w_k10_BP_s00 - w_k10_BP_s002
                                 + 0.99 b_k10_BP_s002 - 1.01 s_k10_BP_s002  = 0
 w_k10_BP_s003_return_constraint: 1.27074 w_k10_BP_s00 - w_k10_BP_s003
                                 + 0.99 b_k10_BP_s003 - 1.01 s_k10_BP_s003  = 0
 w_k10_BP_s004_return_constraint: 1.03443 w_k10_BP_s00 - w_k10_BP_s004
                                 + 0.99 b_k10_BP_s004 - 1.01 s_k10_BP_s004  = 0
 w_k10_BP_s005_return_constraint: 1.11589 w_k10_BP_s00 - w_k10_BP_s005
                                 + 0.99 b_k10_BP_s005 - 1.01 s_k10_BP_s005  = 0
 w_k10_BP_s006_return_constraint: 0.748804 w_k10_BP_s00 - w_k10_BP_s006
                                 + 0.99 b_k10_BP_s006 - 1.01 s_k10_BP_s006  = 0
 w_k10_BP_s007_return_constraint: 1.03116 w_k10_BP_s00 - w_k10_BP_s007
                                 + 0.99 b_k10_BP_s007 - 1.01 s_k10_BP_s007  = 0
 w_k10_BP_s010_return_constraint: 1.36351 w_k10_BP_s01 - w_k10_BP_s010
                                 + 0.99 b_k10_BP_s010 - 1.01 s_k10_BP_s010  = 0
 w_k10_BP_s011_return_constraint: 1.35859 w_k10_BP_s01 - w_k10_BP_s011
                                 + 0.99 b_k10_BP_s011 - 1.01 s_k10_BP_s011  = 0
 w_k10_BP_s012_return_constraint: 0.924343 w_k10_BP_s01 - w_k10_BP_s012
                                 + 0.99 b_k10_BP_s012 - 1.01 s_k10_BP_s012  = 0
 w_k10_BP_s013_return_constraint: 0.769032 w_k10_BP_s01 - w_k10_BP_s013
                                 + 0.99 b_k10_BP_s013 - 1.01 s_k10_BP_s013  = 0
 w_k10_BP_s014_return_constraint: 1.10578 w_k10_BP_s01 - w_k10_BP_s014
                                 + 0.99 b_k10_BP_s014 - 1.01 s_k10_BP_s014  = 0
 w_k10_BP_s015_return_constraint: 1.04402 w_k10_BP_s01 - w_k10_BP_s015
                                 + 0.99 b_k10_BP_s015 - 1.01 s_k10_BP_s015  = 0
 w_k10_BP_s016_return_constraint: 0.696379 w_k10_BP_s01 - w_k10_BP_s016
                                 + 0.99 b_k10_BP_s016 - 1.01 s_k10_BP_s016  = 0
 w_k10_BP_s017_return_constraint: 1.49646 w_k10_BP_s01 - w_k10_BP_s017
                                 + 0.99 b_k10_BP_s017 - 1.01 s_k10_BP_s017  = 0
 w_k10_BP_s100_return_constraint: 1.23524 w_k10_BP_s10 - w_k10_BP_s100
                                 + 0.99 b_k10_BP_s100 - 1.01 s_k10_BP_s100  = 0
 w_k10_BP_s101_return_constraint: 0.737477 w_k10_BP_s10 - w_k10_BP_s101
                                 + 0.99 b_k10_BP_s101 - 1.01 s_k10_BP_s101  = 0
 w_k10_BP_s102_return_constraint: 0.957464 w_k10_BP_s10 - w_k10_BP_s102
                                 + 0.99 b_k10_BP_s102 - 1.01 s_k10_BP_s102  = 0
 w_k10_BP_s103_return_constraint: 0.822475 w_k10_BP_s10 - w_k10_BP_s103
                                 + 0.99 b_k10_BP_s103 - 1.01 s_k10_BP_s103  = 0
 w_k10_BP_s104_return_constraint: 0.920443 w_k10_BP_s10 - w_k10_BP_s104
                                 + 0.99 b_k10_BP_s104 - 1.01 s_k10_BP_s104  = 0
 w_k10_BP_s105_return_constraint: 1.24411 w_k10_BP_s10 - w_k10_BP_s105
                                 + 0.99 b_k10_BP_s105 - 1.01 s_k10_BP_s105  = 0
 w_k10_BP_s106_return_constraint: 1.24174 w_k10_BP_s10 - w_k10_BP_s106
                                 + 0.99 b_k10_BP_s106 - 1.01 s_k10_BP_s106  = 0
 w_k10_BP_s107_return_constraint: 1.12476 w_k10_BP_s10 - w_k10_BP_s107
                                 + 0.99 b_k10_BP_s107 - 1.01 s_k10_BP_s107  = 0
 w_k10_BP_s110_return_constraint: 1.31383 w_k10_BP_s11 - w_k10_BP_s110
                                 + 0.99 b_k10_BP_s110 - 1.01 s_k10_BP_s110  = 0
 w_k10_BP_s111_return_constraint: 0.887305 w_k10_BP_s11 - w_k10_BP_s111
                                 + 0.99 b_k10_BP_s111 - 1.01 s_k10_BP_s111  = 0
 w_k10_BP_s112_return_constraint: 0.809464 w_k10_BP_s11 - w_k10_BP_s112
                                 + 0.99 b_k10_BP_s112 - 1.01 s_k10_BP_s112  = 0
 w_k10_BP_s113_return_constraint: 0.823524 w_k10_BP_s11 - w_k10_BP_s113
                                 + 0.99 b_k10_BP_s113 - 1.01 s_k10_BP_s113  = 0
 w_k10_BP_s114_return_constraint: 1.2094 w_k10_BP_s11 - w_k10_BP_s114
                                 + 0.99 b_k10_BP_s114 - 1.01 s_k10_BP_s114  = 0
 w_k10_BP_s115_return_constraint: 0.738039 w_k10_BP_s11 - w_k10_BP_s115
                                 + 0.99 b_k10_BP_s115 - 1.01 s_k10_BP_s115  = 0
 w_k10_BP_s116_return_constraint: 0.999716 w_k10_BP_s11 - w_k10_BP_s116
                                 + 0.99 b_k10_BP_s116 - 1.01 s_k10_BP_s116  = 0
 w_k10_BP_s117_return_constraint: 1.02985 w_k10_BP_s11 - w_k10_BP_s117
                                 + 0.99 b_k10_BP_s117 - 1.01 s_k10_BP_s117  = 0
 w_k10_GSK_s0_return_constraint: 1.11086 w_0_GSK - w_k10_GSK_s0
                                 + 0.99 b_k10_GSK_s0 - 1.01 s_k10_GSK_s0  = 0
 w_k10_GSK_s1_return_constraint: 0.944089 w_0_GSK - w_k10_GSK_s1
                                 + 0.99 b_k10_GSK_s1 - 1.01 s_k10_GSK_s1  = 0
 w_k10_GSK_s00_return_constraint: 0.966885 w_k10_GSK_s0 - w_k10_GSK_s00
                                 + 0.99 b_k10_GSK_s00 - 1.01 s_k10_GSK_s00  = 0
 w_k10_GSK_s01_return_constraint: 0.954828 w_k10_GSK_s0 - w_k10_GSK_s01
                                 + 0.99 b_k10_GSK_s01 - 1.01 s_k10_GSK_s01  = 0
 w_k10_GSK_s10_return_constraint: 1.01951 w_k10_GSK_s1 - w_k10_GSK_s10
                                 + 0.99 b_k10_GSK_s10 - 1.01 s_k10_GSK_s10  = 0
 w_k10_GSK_s11_return_constraint: 1.24293 w_k10_GSK_s1 - w_k10_GSK_s11
                                 + 0.99 b_k10_GSK_s11 - 1.01 s_k10_GSK_s11  = 0
 w_k10_GSK_s000_return_constraint: 1.38162 w_k10_GSK_s00 - w_k10_GSK_s000
                                 + 0.99 b_k10_GSK_s000 - 1.01 s_k10_GSK_s000
                                  = 0
 w_k10_GSK_s001_return_constraint: 0.858032 w_k10_GSK_s00 - w_k10_GSK_s001
                                 + 0.99 b_k10_GSK_s001 - 1.01 s_k10_GSK_s001
                                  = 0
 w_k10_GSK_s002_return_constraint: 1.03248 w_k10_GSK_s00 - w_k10_GSK_s002
                                 + 0.99 b_k10_GSK_s002 - 1.01 s_k10_GSK_s002
                                  = 0
 w_k10_GSK_s003_return_constraint: 1.11755 w_k10_GSK_s00 - w_k10_GSK_s003
                                 + 0.99 b_k10_GSK_s003 - 1.01 s_k10_GSK_s003
                                  = 0
 w_k10_GSK_s004_return_constraint: 1.10733 w_k10_GSK_s00 - w_k10_GSK_s004
                                 + 0.99 b_k10_GSK_s004 - 1.01 s_k10_GSK_s004
                                  = 0
 w_k10_GSK_s005_return_constraint: 1.07032 w_k10_GSK_s00 - w_k10_GSK_s005
                                 + 0.99 b_k10_GSK_s005 - 1.01 s_k10_GSK_s005
                                  = 0
 w_k10_GSK_s006_return_constraint: 0.903303 w_k10_GSK_s00 - w_k10_GSK_s006
                                 + 0.99 b_k10_GSK_s006 - 1.01 s_k10_GSK_s006
                                  = 0
 w_k10_GSK_s007_return_constraint: 0.971542 w_k10_GSK_s00 - w_k10_GSK_s007
                                 + 0.99 b_k10_GSK_s007 - 1.01 s_k10_GSK_s007
                                  = 0
 w_k10_GSK_s010_return_constraint: 1.19852 w_k10_GSK_s01 - w_k10_GSK_s010
                                 + 0.99 b_k10_GSK_s010 - 1.01 s_k10_GSK_s010
                                  = 0
 w_k10_GSK_s011_return_constraint: 1.22968 w_k10_GSK_s01 - w_k10_GSK_s011
                                 + 0.99 b_k10_GSK_s011 - 1.01 s_k10_GSK_s011
                                  = 0
 w_k10_GSK_s012_return_constraint: 0.945077 w_k10_GSK_s01 - w_k10_GSK_s012
                                 + 0.99 b_k10_GSK_s012 - 1.01 s_k10_GSK_s012
                                  = 0
 w_k10_GSK_s013_return_constraint: 1.00898 w_k10_GSK_s01 - w_k10_GSK_s013
                                 + 0.99 b_k10_GSK_s013 - 1.01 s_k10_GSK_s013
                                  = 0
 w_k10_GSK_s014_return_constraint: 1.09338 w_k10_GSK_s01 - w_k10_GSK_s014
                                 + 0.99 b_k10_GSK_s014 - 1.01 s_k10_GSK_s014
                                  = 0
 w_k10_GSK_s015_return_constraint: 0.953531 w_k10_GSK_s01 - w_k10_GSK_s015
                                 + 0.99 b_k10_GSK_s015 - 1.01 s_k10_GSK_s015
                                  = 0
 w_k10_GSK_s016_return_constraint: 0.853662 w_k10_GSK_s01 - w_k10_GSK_s016
                                 + 0.99 b_k10_GSK_s016 - 1.01 s_k10_GSK_s016
                                  = 0
 w_k10_GSK_s017_return_constraint: 1.37392 w_k10_GSK_s01 - w_k10_GSK_s017
                                 + 0.99 b_k10_GSK_s017 - 1.01 s_k10_GSK_s017
                                  = 0
 w_k10_GSK_s100_return_constraint: 1.09663 w_k10_GSK_s10 - w_k10_GSK_s100
                                 + 0.99 b_k10_GSK_s100 - 1.01 s_k10_GSK_s100
                                  = 0
 w_k10_GSK_s101_return_constraint: 0.861458 w_k10_GSK_s10 - w_k10_GSK_s101
                                 + 0.99 b_k10_GSK_s101 - 1.01 s_k10_GSK_s101
                                  = 0
 w_k10_GSK_s102_return_constraint: 0.928904 w_k10_GSK_s10 - w_k10_GSK_s102
                                 + 0.99 b_k10_GSK_s102 - 1.01 s_k10_GSK_s102
                                  = 0
 w_k10_GSK_s103_return_constraint: 0.81049 w_k10_GSK_s10 - w_k10_GSK_s103
                                 + 0.99 b_k10_GSK_s103 - 1.01 s_k10_GSK_s103
                                  = 0
 w_k10_GSK_s104_return_constraint: 0.946648 w_k10_GSK_s10 - w_k10_GSK_s104
                                 + 0.99 b_k10_GSK_s104 - 1.01 s_k10_GSK_s104
                                  = 0
 w_k10_GSK_s105_return_constraint: 1.06841 w_k10_GSK_s10 - w_k10_GSK_s105
                                 + 0.99 b_k10_GSK_s105 - 1.01 s_k10_GSK_s105
                                  = 0
 w_k10_GSK_s106_return_constraint: 1.01504 w_k10_GSK_s10 - w_k10_GSK_s106
                                 + 0.99 b_k10_GSK_s106 - 1.01 s_k10_GSK_s106
                                  = 0
 w_k10_GSK_s107_return_constraint: 0.980438 w_k10_GSK_s10 - w_k10_GSK_s107
                                 + 0.99 b_k10_GSK_s107 - 1.01 s_k10_GSK_s107
                                  = 0
 w_k10_GSK_s110_return_constraint: 1.15521 w_k10_GSK_s11 - w_k10_GSK_s110
                                 + 0.99 b_k10_GSK_s110 - 1.01 s_k10_GSK_s110
                                  = 0
 w_k10_GSK_s111_return_constraint: 0.88154 w_k10_GSK_s11 - w_k10_GSK_s111
                                 + 0.99 b_k10_GSK_s111 - 1.01 s_k10_GSK_s111
                                  = 0
 w_k10_GSK_s112_return_constraint: 0.89919 w_k10_GSK_s11 - w_k10_GSK_s112
                                 + 0.99 b_k10_GSK_s112 - 1.01 s_k10_GSK_s112
                                  = 0
 w_k10_GSK_s113_return_constraint: 0.931381 w_k10_GSK_s11 - w_k10_GSK_s113
                                 + 0.99 b_k10_GSK_s113 - 1.01 s_k10_GSK_s113
                                  = 0
 w_k10_GSK_s114_return_constraint: 1.05491 w_k10_GSK_s11 - w_k10_GSK_s114
                                 + 0.99 b_k10_GSK_s114 - 1.01 s_k10_GSK_s114
                                  = 0
 w_k10_GSK_s115_return_constraint: 0.774055 w_k10_GSK_s11 - w_k10_GSK_s115
                                 + 0.99 b_k10_GSK_s115 - 1.01 s_k10_GSK_s115
                                  = 0
 w_k10_GSK_s116_return_constraint: 0.957032 w_k10_GSK_s11 - w_k10_GSK_s116
                                 + 0.99 b_k10_GSK_s116 - 1.01 s_k10_GSK_s116
                                  = 0
 w_k10_GSK_s117_return_constraint: 0.972945 w_k10_GSK_s11 - w_k10_GSK_s117
                                 + 0.99 b_k10_GSK_s117 - 1.01 s_k10_GSK_s117
                                  = 0
 w_k10_AZN_s0_return_constraint: 0.999203 w_0_AZN - w_k10_AZN_s0
                                 + 0.99 b_k10_AZN_s0 - 1.01 s_k10_AZN_s0  = 0
 w_k10_AZN_s1_return_constraint: 0.984172 w_0_AZN - w_k10_AZN_s1
                                 + 0.99 b_k10_AZN_s1 - 1.01 s_k10_AZN_s1  = 0
 w_k10_AZN_s00_return_constraint: 1.06744 w_k10_AZN_s0 - w_k10_AZN_s00
                                 + 0.99 b_k10_AZN_s00 - 1.01 s_k10_AZN_s00  = 0
 w_k10_AZN_s01_return_constraint: 1.03396 w_k10_AZN_s0 - w_k10_AZN_s01
                                 + 0.99 b_k10_AZN_s01 - 1.01 s_k10_AZN_s01  = 0
 w_k10_AZN_s10_return_constraint: 0.998657 w_k10_AZN_s1 - w_k10_AZN_s10
                                 + 0.99 b_k10_AZN_s10 - 1.01 s_k10_AZN_s10  = 0
 w_k10_AZN_s11_return_constraint: 1.07696 w_k10_AZN_s1 - w_k10_AZN_s11
                                 + 0.99 b_k10_AZN_s11 - 1.01 s_k10_AZN_s11  = 0
 w_k10_AZN_s000_return_constraint: 1.04645 w_k10_AZN_s00 - w_k10_AZN_s000
                                 + 0.99 b_k10_AZN_s000 - 1.01 s_k10_AZN_s000
                                  = 0
 w_k10_AZN_s001_return_constraint: 0.925556 w_k10_AZN_s00 - w_k10_AZN_s001
                                 + 0.99 b_k10_AZN_s001 - 1.01 s_k10_AZN_s001
                                  = 0
 w_k10_AZN_s002_return_constraint: 1.01999 w_k10_AZN_s00 - w_k10_AZN_s002
                                 + 0.99 b_k10_AZN_s002 - 1.01 s_k10_AZN_s002
                                  = 0
 w_k10_AZN_s003_return_constraint: 0.961565 w_k10_AZN_s00 - w_k10_AZN_s003
                                 + 0.99 b_k10_AZN_s003 - 1.01 s_k10_AZN_s003
                                  = 0
 w_k10_AZN_s004_return_constraint: 1.01551 w_k10_AZN_s00 - w_k10_AZN_s004
                                 + 0.99 b_k10_AZN_s004 - 1.01 s_k10_AZN_s004
                                  = 0
 w_k10_AZN_s005_return_constraint: 1.1419 w_k10_AZN_s00 - w_k10_AZN_s005
                                 + 0.99 b_k10_AZN_s005 - 1.01 s_k10_AZN_s005
                                  = 0
 w_k10_AZN_s006_return_constraint: 0.937728 w_k10_AZN_s00 - w_k10_AZN_s006
                                 + 0.99 b_k10_AZN_s006 - 1.01 s_k10_AZN_s006
                                  = 0
 w_k10_AZN_s007_return_constraint: 1.0328 w_k10_AZN_s00 - w_k10_AZN_s007
                                 + 0.99 b_k10_AZN_s007 - 1.01 s_k10_AZN_s007
                                  = 0
 w_k10_AZN_s010_return_constraint: 0.98009 w_k10_AZN_s01 - w_k10_AZN_s010
                                 + 0.99 b_k10_AZN_s010 - 1.01 s_k10_AZN_s010
                                  = 0
 w_k10_AZN_s011_return_constraint: 1.04631 w_k10_AZN_s01 - w_k10_AZN_s011
                                 + 0.99 b_k10_AZN_s011 - 1.01 s_k10_AZN_s011
                                  = 0
 w_k10_AZN_s012_return_constraint: 0.960939 w_k10_AZN_s01 - w_k10_AZN_s012
                                 + 0.99 b_k10_AZN_s012 - 1.01 s_k10_AZN_s012
                                  = 0
 w_k10_AZN_s013_return_constraint: 1.02128 w_k10_AZN_s01 - w_k10_AZN_s013
                                 + 0.99 b_k10_AZN_s013 - 1.01 s_k10_AZN_s013
                                  = 0
 w_k10_AZN_s014_return_constraint: 1.04214 w_k10_AZN_s01 - w_k10_AZN_s014
                                 + 0.99 b_k10_AZN_s014 - 1.01 s_k10_AZN_s014
                                  = 0
 w_k10_AZN_s015_return_constraint: 0.938562 w_k10_AZN_s01 - w_k10_AZN_s015
                                 + 0.99 b_k10_AZN_s015 - 1.01 s_k10_AZN_s015
                                  = 0
 w_k10_AZN_s016_return_constraint: 0.932712 w_k10_AZN_s01 - w_k10_AZN_s016
                                 + 0.99 b_k10_AZN_s016 - 1.01 s_k10_AZN_s016
                                  = 0
 w_k10_AZN_s017_return_constraint: 1.11699 w_k10_AZN_s01 - w_k10_AZN_s017
                                 + 0.99 b_k10_AZN_s017 - 1.01 s_k10_AZN_s017
                                  = 0
 w_k10_AZN_s100_return_constraint: 1.04033 w_k10_AZN_s10 - w_k10_AZN_s100
                                 + 0.99 b_k10_AZN_s100 - 1.01 s_k10_AZN_s100
                                  = 0
 w_k10_AZN_s101_return_constraint: 0.925819 w_k10_AZN_s10 - w_k10_AZN_s101
                                 + 0.99 b_k10_AZN_s101 - 1.01 s_k10_AZN_s101
                                  = 0
 w_k10_AZN_s102_return_constraint: 0.93367 w_k10_AZN_s10 - w_k10_AZN_s102
                                 + 0.99 b_k10_AZN_s102 - 1.01 s_k10_AZN_s102
                                  = 0
 w_k10_AZN_s103_return_constraint: 0.952443 w_k10_AZN_s10 - w_k10_AZN_s103
                                 + 0.99 b_k10_AZN_s103 - 1.01 s_k10_AZN_s103
                                  = 0
 w_k10_AZN_s104_return_constraint: 0.908367 w_k10_AZN_s10 - w_k10_AZN_s104
                                 + 0.99 b_k10_AZN_s104 - 1.01 s_k10_AZN_s104
                                  = 0
 w_k10_AZN_s105_return_constraint: 0.979594 w_k10_AZN_s10 - w_k10_AZN_s105
                                 + 0.99 b_k10_AZN_s105 - 1.01 s_k10_AZN_s105
                                  = 0
 w_k10_AZN_s106_return_constraint: 1.02462 w_k10_AZN_s10 - w_k10_AZN_s106
                                 + 0.99 b_k10_AZN_s106 - 1.01 s_k10_AZN_s106
                                  = 0
 w_k10_AZN_s107_return_constraint: 1.01299 w_k10_AZN_s10 - w_k10_AZN_s107
                                 + 0.99 b_k10_AZN_s107 - 1.01 s_k10_AZN_s107
                                  = 0
 w_k10_AZN_s110_return_constraint: 1.21739 w_k10_AZN_s11 - w_k10_AZN_s110
                                 + 0.99 b_k10_AZN_s110 - 1.01 s_k10_AZN_s110
                                  = 0
 w_k10_AZN_s111_return_constraint: 1.00075 w_k10_AZN_s11 - w_k10_AZN_s111
                                 + 0.99 b_k10_AZN_s111 - 1.01 s_k10_AZN_s111
                                  = 0
 w_k10_AZN_s112_return_constraint: 0.98846 w_k10_AZN_s11 - w_k10_AZN_s112
                                 + 0.99 b_k10_AZN_s112 - 1.01 s_k10_AZN_s112
                                  = 0
 w_k10_AZN_s113_return_constraint: 1.02296 w_k10_AZN_s11 - w_k10_AZN_s113
                                 + 0.99 b_k10_AZN_s113 - 1.01 s_k10_AZN_s113
                                  = 0
 w_k10_AZN_s114_return_constraint: 0.871939 w_k10_AZN_s11 - w_k10_AZN_s114
                                 + 0.99 b_k10_AZN_s114 - 1.01 s_k10_AZN_s114
                                  = 0
 w_k10_AZN_s115_return_constraint: 0.985865 w_k10_AZN_s11 - w_k10_AZN_s115
                                 + 0.99 b_k10_AZN_s115 - 1.01 s_k10_AZN_s115
                                  = 0
 w_k10_AZN_s116_return_constraint: 1.04141 w_k10_AZN_s11 - w_k10_AZN_s116
                                 + 0.99 b_k10_AZN_s116 - 1.01 s_k10_AZN_s116
                                  = 0
 w_k10_AZN_s117_return_constraint: 1.10908 w_k10_AZN_s11 - w_k10_AZN_s117
                                 + 0.99 b_k10_AZN_s117 - 1.01 s_k10_AZN_s117
                                  = 0
 w_k10_RIO_s0_return_constraint: 1.10477 w_0_RIO - w_k10_RIO_s0
                                 + 0.99 b_k10_RIO_s0 - 1.01 s_k10_RIO_s0  = 0
 w_k10_RIO_s1_return_constraint: 0.921553 w_0_RIO - w_k10_RIO_s1
                                 + 0.99 b_k10_RIO_s1 - 1.01 s_k10_RIO_s1  = 0
 w_k10_RIO_s00_return_constraint: 1.01078 w_k10_RIO_s0 - w_k10_RIO_s00
                                 + 0.99 b_k10_RIO_s00 - 1.01 s_k10_RIO_s00  = 0
 w_k10_RIO_s01_return_constraint: 1.02965 w_k10_RIO_s0 - w_k10_RIO_s01
                                 + 0.99 b_k10_RIO_s01 - 1.01 s_k10_RIO_s01  = 0
 w_k10_RIO_s10_return_constraint: 0.932568 w_k10_RIO_s1 - w_k10_RIO_s10
                                 + 0.99 b_k10_RIO_s10 - 1.01 s_k10_RIO_s10  = 0
 w_k10_RIO_s11_return_constraint: 1.21134 w_k10_RIO_s1 - w_k10_RIO_s11
                                 + 0.99 b_k10_RIO_s11 - 1.01 s_k10_RIO_s11  = 0
 w_k10_RIO_s000_return_constraint: 1.37286 w_k10_RIO_s00 - w_k10_RIO_s000
                                 + 0.99 b_k10_RIO_s000 - 1.01 s_k10_RIO_s000
                                  = 0
 w_k10_RIO_s001_return_constraint: 0.75844 w_k10_RIO_s00 - w_k10_RIO_s001
                                 + 0.99 b_k10_RIO_s001 - 1.01 s_k10_RIO_s001
                                  = 0
 w_k10_RIO_s002_return_constraint: 0.883828 w_k10_RIO_s00 - w_k10_RIO_s002
                                 + 0.99 b_k10_RIO_s002 - 1.01 s_k10_RIO_s002
                                  = 0
 w_k10_RIO_s003_return_constraint: 1.23416 w_k10_RIO_s00 - w_k10_RIO_s003
                                 + 0.99 b_k10_RIO_s003 - 1.01 s_k10_RIO_s003
                                  = 0
 w_k10_RIO_s004_return_constraint: 1.04759 w_k10_RIO_s00 - w_k10_RIO_s004
                                 + 0.99 b_k10_RIO_s004 - 1.01 s_k10_RIO_s004
                                  = 0
 w_k10_RIO_s005_return_constraint: 0.995813 w_k10_RIO_s00 - w_k10_RIO_s005
                                 + 0.99 b_k10_RIO_s005 - 1.01 s_k10_RIO_s005
                                  = 0
 w_k10_RIO_s006_return_constraint: 0.780048 w_k10_RIO_s00 - w_k10_RIO_s006
                                 + 0.99 b_k10_RIO_s006 - 1.01 s_k10_RIO_s006
                                  = 0
 w_k10_RIO_s007_return_constraint: 1.04187 w_k10_RIO_s00 - w_k10_RIO_s007
                                 + 0.99 b_k10_RIO_s007 - 1.01 s_k10_RIO_s007
                                  = 0
 w_k10_RIO_s010_return_constraint: 1.39312 w_k10_RIO_s01 - w_k10_RIO_s010
                                 + 0.99 b_k10_RIO_s010 - 1.01 s_k10_RIO_s010
                                  = 0
 w_k10_RIO_s011_return_constraint: 1.21268 w_k10_RIO_s01 - w_k10_RIO_s011
                                 + 0.99 b_k10_RIO_s011 - 1.01 s_k10_RIO_s011
                                  = 0
 w_k10_RIO_s012_return_constraint: 0.984758 w_k10_RIO_s01 - w_k10_RIO_s012
                                 + 0.99 b_k10_RIO_s012 - 1.01 s_k10_RIO_s012
                                  = 0
 w_k10_RIO_s013_return_constraint: 0.852018 w_k10_RIO_s01 - w_k10_RIO_s013
                                 + 0.99 b_k10_RIO_s013 - 1.01 s_k10_RIO_s013
                                  = 0
 w_k10_RIO_s014_return_constraint: 1.11136 w_k10_RIO_s01 - w_k10_RIO_s014
                                 + 0.99 b_k10_RIO_s014 - 1.01 s_k10_RIO_s014
                                  = 0
 w_k10_RIO_s015_return_constraint: 1.02574 w_k10_RIO_s01 - w_k10_RIO_s015
                                 + 0.99 b_k10_RIO_s015 - 1.01 s_k10_RIO_s015
                                  = 0
 w_k10_RIO_s016_return_constraint: 0.727927 w_k10_RIO_s01 - w_k10_RIO_s016
                                 + 0.99 b_k10_RIO_s016 - 1.01 s_k10_RIO_s016
                                  = 0
 w_k10_RIO_s017_return_constraint: 1.35097 w_k10_RIO_s01 - w_k10_RIO_s017
                                 + 0.99 b_k10_RIO_s017 - 1.01 s_k10_RIO_s017
                                  = 0
 w_k10_RIO_s100_return_constraint: 1.24413 w_k10_RIO_s10 - w_k10_RIO_s100
                                 + 0.99 b_k10_RIO_s100 - 1.01 s_k10_RIO_s100
                                  = 0
 w_k10_RIO_s101_return_constraint: 0.809815 w_k10_RIO_s10 - w_k10_RIO_s101
                                 + 0.99 b_k10_RIO_s101 - 1.01 s_k10_RIO_s101
                                  = 0
 w_k10_RIO_s102_return_constraint: 0.928067 w_k10_RIO_s10 - w_k10_RIO_s102
                                 + 0.99 b_k10_RIO_s102 - 1.01 s_k10_RIO_s102
                                  = 0
 w_k10_RIO_s103_return_constraint: 0.835033 w_k10_RIO_s10 - w_k10_RIO_s103
                                 + 0.99 b_k10_RIO_s103 - 1.01 s_k10_RIO_s103
                                  = 0
 w_k10_RIO_s104_return_constraint: 0.987953 w_k10_RIO_s10 - w_k10_RIO_s104
                                 + 0.99 b_k10_RIO_s104 - 1.01 s_k10_RIO_s104
                                  = 0
 w_k10_RIO_s105_return_constraint: 1.31172 w_k10_RIO_s10 - w_k10_RIO_s105
                                 + 0.99 b_k10_RIO_s105 - 1.01 s_k10_RIO_s105
                                  = 0
 w_k10_RIO_s106_return_constraint: 1.23237 w_k10_RIO_s10 - w_k10_RIO_s106
                                 + 0.99 b_k10_RIO_s106 - 1.01 s_k10_RIO_s106
                                  = 0
 w_k10_RIO_s107_return_constraint: 1.07718 w_k10_RIO_s10 - w_k10_RIO_s107
                                 + 0.99 b_k10_RIO_s107 - 1.01 s_k10_RIO_s107
                                  = 0
 w_k10_RIO_s110_return_constraint: 1.27997 w_k10_RIO_s11 - w_k10_RIO_s110
                                 + 0.99 b_k10_RIO_s110 - 1.01 s_k10_RIO_s110
                                  = 0
 w_k10_RIO_s111_return_constraint: 0.958968 w_k10_RIO_s11 - w_k10_RIO_s111
                                 + 0.99 b_k10_RIO_s111 - 1.01 s_k10_RIO_s111
                                  = 0
 w_k10_RIO_s112_return_constraint: 0.852921 w_k10_RIO_s11 - w_k10_RIO_s112
                                 + 0.99 b_k10_RIO_s112 - 1.01 s_k10_RIO_s112
                                  = 0
 w_k10_RIO_s113_return_constraint: 0.91346 w_k10_RIO_s11 - w_k10_RIO_s113
                                 + 0.99 b_k10_RIO_s113 - 1.01 s_k10_RIO_s113
                                  = 0
 w_k10_RIO_s114_return_constraint: 1.27326 w_k10_RIO_s11 - w_k10_RIO_s114
                                 + 0.99 b_k10_RIO_s114 - 1.01 s_k10_RIO_s114
                                  = 0
 w_k10_RIO_s115_return_constraint: 0.743365 w_k10_RIO_s11 - w_k10_RIO_s115
                                 + 0.99 b_k10_RIO_s115 - 1.01 s_k10_RIO_s115
                                  = 0
 w_k10_RIO_s116_return_constraint: 1.06653 w_k10_RIO_s11 - w_k10_RIO_s116
                                 + 0.99 b_k10_RIO_s116 - 1.01 s_k10_RIO_s116
                                  = 0
 w_k10_RIO_s117_return_constraint: 0.992415 w_k10_RIO_s11 - w_k10_RIO_s117
                                 + 0.99 b_k10_RIO_s117 - 1.01 s_k10_RIO_s117
                                  = 0
 w_k10_BG_s0_return_constraint:  1.0497 w_0_BG - w_k10_BG_s0 + 0.99 b_k10_BG_s0
                                 - 1.01 s_k10_BG_s0  = 0
 w_k10_BG_s1_return_constraint:  0.972592 w_0_BG - w_k10_BG_s1
                                 + 0.99 b_k10_BG_s1 - 1.01 s_k10_BG_s1  = 0
 w_k10_BG_s00_return_constraint: 0.922479 w_k10_BG_s0 - w_k10_BG_s00
                                 + 0.99 b_k10_BG_s00 - 1.01 s_k10_BG_s00  = 0
 w_k10_BG_s01_return_constraint: 1.06321 w_k10_BG_s0 - w_k10_BG_s01
                                 + 0.99 b_k10_BG_s01 - 1.01 s_k10_BG_s01  = 0
 w_k10_BG_s10_return_constraint: 0.947816 w_k10_BG_s1 - w_k10_BG_s10
                                 + 0.99 b_k10_BG_s10 - 1.01 s_k10_BG_s10  = 0
 w_k10_BG_s11_return_constraint: 1.0513 w_k10_BG_s1 - w_k10_BG_s11
                                 + 0.99 b_k10_BG_s11 - 1.01 s_k10_BG_s11  = 0
 w_k10_BG_s000_return_constraint: 1.11195 w_k10_BG_s00 - w_k10_BG_s000
                                 + 0.99 b_k10_BG_s000 - 1.01 s_k10_BG_s000  = 0
 w_k10_BG_s001_return_constraint: 0.90731 w_k10_BG_s00 - w_k10_BG_s001
                                 + 0.99 b_k10_BG_s001 - 1.01 s_k10_BG_s001  = 0
 w_k10_BG_s002_return_constraint: 1.02108 w_k10_BG_s00 - w_k10_BG_s002
                                 + 0.99 b_k10_BG_s002 - 1.01 s_k10_BG_s002  = 0
 w_k10_BG_s003_return_constraint: 1.157 w_k10_BG_s00 - w_k10_BG_s003
                                 + 0.99 b_k10_BG_s003 - 1.01 s_k10_BG_s003  = 0
 w_k10_BG_s004_return_constraint: 1.0338 w_k10_BG_s00 - w_k10_BG_s004
                                 + 0.99 b_k10_BG_s004 - 1.01 s_k10_BG_s004  = 0
 w_k10_BG_s005_return_constraint: 0.973476 w_k10_BG_s00 - w_k10_BG_s005
                                 + 0.99 b_k10_BG_s005 - 1.01 s_k10_BG_s005  = 0
 w_k10_BG_s006_return_constraint: 0.980258 w_k10_BG_s00 - w_k10_BG_s006
                                 + 0.99 b_k10_BG_s006 - 1.01 s_k10_BG_s006  = 0
 w_k10_BG_s007_return_constraint: 0.9784 w_k10_BG_s00 - w_k10_BG_s007
                                 + 0.99 b_k10_BG_s007 - 1.01 s_k10_BG_s007  = 0
 w_k10_BG_s010_return_constraint: 1.15754 w_k10_BG_s01 - w_k10_BG_s010
                                 + 0.99 b_k10_BG_s010 - 1.01 s_k10_BG_s010  = 0
 w_k10_BG_s011_return_constraint: 1.00312 w_k10_BG_s01 - w_k10_BG_s011
                                 + 0.99 b_k10_BG_s011 - 1.01 s_k10_BG_s011  = 0
 w_k10_BG_s012_return_constraint: 0.953502 w_k10_BG_s01 - w_k10_BG_s012
                                 + 0.99 b_k10_BG_s012 - 1.01 s_k10_BG_s012  = 0
 w_k10_BG_s013_return_constraint: 0.848929 w_k10_BG_s01 - w_k10_BG_s013
                                 + 0.99 b_k10_BG_s013 - 1.01 s_k10_BG_s013  = 0
 w_k10_BG_s014_return_constraint: 1.04087 w_k10_BG_s01 - w_k10_BG_s014
                                 + 0.99 b_k10_BG_s014 - 1.01 s_k10_BG_s014  = 0
 w_k10_BG_s015_return_constraint: 1.01448 w_k10_BG_s01 - w_k10_BG_s015
                                 + 0.99 b_k10_BG_s015 - 1.01 s_k10_BG_s015  = 0
 w_k10_BG_s016_return_constraint: 0.955622 w_k10_BG_s01 - w_k10_BG_s016
                                 + 0.99 b_k10_BG_s016 - 1.01 s_k10_BG_s016  = 0
 w_k10_BG_s017_return_constraint: 1.14465 w_k10_BG_s01 - w_k10_BG_s017
                                 + 0.99 b_k10_BG_s017 - 1.01 s_k10_BG_s017  = 0
 w_k10_BG_s100_return_constraint: 1.00174 w_k10_BG_s10 - w_k10_BG_s100
                                 + 0.99 b_k10_BG_s100 - 1.01 s_k10_BG_s100  = 0
 w_k10_BG_s101_return_constraint: 0.949775 w_k10_BG_s10 - w_k10_BG_s101
                                 + 0.99 b_k10_BG_s101 - 1.01 s_k10_BG_s101  = 0
 w_k10_BG_s102_return_constraint: 1.0361 w_k10_BG_s10 - w_k10_BG_s102
                                 + 0.99 b_k10_BG_s102 - 1.01 s_k10_BG_s102  = 0
 w_k10_BG_s103_return_constraint: 0.856212 w_k10_BG_s10 - w_k10_BG_s103
                                 + 0.99 b_k10_BG_s103 - 1.01 s_k10_BG_s103  = 0
 w_k10_BG_s104_return_constraint: 1.16809 w_k10_BG_s10 - w_k10_BG_s104
                                 + 0.99 b_k10_BG_s104 - 1.01 s_k10_BG_s104  = 0
 w_k10_BG_s105_return_constraint: 1.17209 w_k10_BG_s10 - w_k10_BG_s105
                                 + 0.99 b_k10_BG_s105 - 1.01 s_k10_BG_s105  = 0
 w_k10_BG_s106_return_constraint: 1.11145 w_k10_BG_s10 - w_k10_BG_s106
                                 + 0.99 b_k10_BG_s106 - 1.01 s_k10_BG_s106  = 0
 w_k10_BG_s107_return_constraint: 0.895463 w_k10_BG_s10 - w_k10_BG_s107
                                 + 0.99 b_k10_BG_s107 - 1.01 s_k10_BG_s107  = 0
 w_k10_BG_s110_return_constraint: 0.992194 w_k10_BG_s11 - w_k10_BG_s110
                                 + 0.99 b_k10_BG_s110 - 1.01 s_k10_BG_s110  = 0
 w_k10_BG_s111_return_constraint: 1.03248 w_k10_BG_s11 - w_k10_BG_s111
                                 + 0.99 b_k10_BG_s111 - 1.01 s_k10_BG_s111  = 0
 w_k10_BG_s112_return_constraint: 1.07826 w_k10_BG_s11 - w_k10_BG_s112
                                 + 0.99 b_k10_BG_s112 - 1.01 s_k10_BG_s112  = 0
 w_k10_BG_s113_return_constraint: 0.966047 w_k10_BG_s11 - w_k10_BG_s113
                                 + 0.99 b_k10_BG_s113 - 1.01 s_k10_BG_s113  = 0
 w_k10_BG_s114_return_constraint: 1.12666 w_k10_BG_s11 - w_k10_BG_s114
                                 + 0.99 b_k10_BG_s114 - 1.01 s_k10_BG_s114  = 0
 w_k10_BG_s115_return_constraint: 0.874823 w_k10_BG_s11 - w_k10_BG_s115
                                 + 0.99 b_k10_BG_s115 - 1.01 s_k10_BG_s115  = 0
 w_k10_BG_s116_return_constraint: 0.969596 w_k10_BG_s11 - w_k10_BG_s116
                                 + 0.99 b_k10_BG_s116 - 1.01 s_k10_BG_s116  = 0
 w_k10_BG_s117_return_constraint: 0.968169 w_k10_BG_s11 - w_k10_BG_s117
                                 + 0.99 b_k10_BG_s117 - 1.01 s_k10_BG_s117  = 0
 w_k10_TSCO_s0_return_constraint: 1.15621 w_0_TSCO - w_k10_TSCO_s0
                                 + 0.99 b_k10_TSCO_s0 - 1.01 s_k10_TSCO_s0  = 0
 w_k10_TSCO_s1_return_constraint: 0.940908 w_0_TSCO - w_k10_TSCO_s1
                                 + 0.99 b_k10_TSCO_s1 - 1.01 s_k10_TSCO_s1  = 0
 w_k10_TSCO_s00_return_constraint: 0.992779 w_k10_TSCO_s0 - w_k10_TSCO_s00
                                 + 0.99 b_k10_TSCO_s00 - 1.01 s_k10_TSCO_s00
                                  = 0
 w_k10_TSCO_s01_return_constraint: 1.0238 w_k10_TSCO_s0 - w_k10_TSCO_s01
                                 + 0.99 b_k10_TSCO_s01 - 1.01 s_k10_TSCO_s01
                                  = 0
 w_k10_TSCO_s10_return_constraint: 0.937347 w_k10_TSCO_s1 - w_k10_TSCO_s10
                                 + 0.99 b_k10_TSCO_s10 - 1.01 s_k10_TSCO_s10
                                  = 0
 w_k10_TSCO_s11_return_constraint: 1.23517 w_k10_TSCO_s1 - w_k10_TSCO_s11
                                 + 0.99 b_k10_TSCO_s11 - 1.01 s_k10_TSCO_s11
                                  = 0
 w_k10_TSCO_s000_return_constraint: 1.61734 w_k10_TSCO_s00 - w_k10_TSCO_s000
                                 + 0.99 b_k10_TSCO_s000 - 1.01 s_k10_TSCO_s000
                                  = 0
 w_k10_TSCO_s001_return_constraint: 0.743656 w_k10_TSCO_s00 - w_k10_TSCO_s001
                                 + 0.99 b_k10_TSCO_s001 - 1.01 s_k10_TSCO_s001
                                  = 0
 w_k10_TSCO_s002_return_constraint: 0.97315 w_k10_TSCO_s00 - w_k10_TSCO_s002
                                 + 0.99 b_k10_TSCO_s002 - 1.01 s_k10_TSCO_s002
                                  = 0
 w_k10_TSCO_s003_return_constraint: 1.15627 w_k10_TSCO_s00 - w_k10_TSCO_s003
                                 + 0.99 b_k10_TSCO_s003 - 1.01 s_k10_TSCO_s003
                                  = 0
 w_k10_TSCO_s004_return_constraint: 1.02145 w_k10_TSCO_s00 - w_k10_TSCO_s004
                                 + 0.99 b_k10_TSCO_s004 - 1.01 s_k10_TSCO_s004
                                  = 0
 w_k10_TSCO_s005_return_constraint: 0.945619 w_k10_TSCO_s00 - w_k10_TSCO_s005
                                 + 0.99 b_k10_TSCO_s005 - 1.01 s_k10_TSCO_s005
                                  = 0
 w_k10_TSCO_s006_return_constraint: 0.738618 w_k10_TSCO_s00 - w_k10_TSCO_s006
                                 + 0.99 b_k10_TSCO_s006 - 1.01 s_k10_TSCO_s006
                                  = 0
 w_k10_TSCO_s007_return_constraint: 1.02028 w_k10_TSCO_s00 - w_k10_TSCO_s007
                                 + 0.99 b_k10_TSCO_s007 - 1.01 s_k10_TSCO_s007
                                  = 0
 w_k10_TSCO_s010_return_constraint: 1.31077 w_k10_TSCO_s01 - w_k10_TSCO_s010
                                 + 0.99 b_k10_TSCO_s010 - 1.01 s_k10_TSCO_s010
                                  = 0
 w_k10_TSCO_s011_return_constraint: 1.45055 w_k10_TSCO_s01 - w_k10_TSCO_s011
                                 + 0.99 b_k10_TSCO_s011 - 1.01 s_k10_TSCO_s011
                                  = 0
 w_k10_TSCO_s012_return_constraint: 0.884992 w_k10_TSCO_s01 - w_k10_TSCO_s012
                                 + 0.99 b_k10_TSCO_s012 - 1.01 s_k10_TSCO_s012
                                  = 0
 w_k10_TSCO_s013_return_constraint: 0.764406 w_k10_TSCO_s01 - w_k10_TSCO_s013
                                 + 0.99 b_k10_TSCO_s013 - 1.01 s_k10_TSCO_s013
                                  = 0
 w_k10_TSCO_s014_return_constraint: 1.14511 w_k10_TSCO_s01 - w_k10_TSCO_s014
                                 + 0.99 b_k10_TSCO_s014 - 1.01 s_k10_TSCO_s014
                                  = 0
 w_k10_TSCO_s015_return_constraint: 0.943297 w_k10_TSCO_s01 - w_k10_TSCO_s015
                                 + 0.99 b_k10_TSCO_s015 - 1.01 s_k10_TSCO_s015
                                  = 0
 w_k10_TSCO_s016_return_constraint: 0.706058 w_k10_TSCO_s01 - w_k10_TSCO_s016
                                 + 0.99 b_k10_TSCO_s016 - 1.01 s_k10_TSCO_s016
                                  = 0
 w_k10_TSCO_s017_return_constraint: 1.37412 w_k10_TSCO_s01 - w_k10_TSCO_s017
                                 + 0.99 b_k10_TSCO_s017 - 1.01 s_k10_TSCO_s017
                                  = 0
 w_k10_TSCO_s100_return_constraint: 1.10387 w_k10_TSCO_s10 - w_k10_TSCO_s100
                                 + 0.99 b_k10_TSCO_s100 - 1.01 s_k10_TSCO_s100
                                  = 0
 w_k10_TSCO_s101_return_constraint: 0.721201 w_k10_TSCO_s10 - w_k10_TSCO_s101
                                 + 0.99 b_k10_TSCO_s101 - 1.01 s_k10_TSCO_s101
                                  = 0
 w_k10_TSCO_s102_return_constraint: 0.944905 w_k10_TSCO_s10 - w_k10_TSCO_s102
                                 + 0.99 b_k10_TSCO_s102 - 1.01 s_k10_TSCO_s102
                                  = 0
 w_k10_TSCO_s103_return_constraint: 0.822406 w_k10_TSCO_s10 - w_k10_TSCO_s103
                                 + 0.99 b_k10_TSCO_s103 - 1.01 s_k10_TSCO_s103
                                  = 0
 w_k10_TSCO_s104_return_constraint: 0.876759 w_k10_TSCO_s10 - w_k10_TSCO_s104
                                 + 0.99 b_k10_TSCO_s104 - 1.01 s_k10_TSCO_s104
                                  = 0
 w_k10_TSCO_s105_return_constraint: 1.3462 w_k10_TSCO_s10 - w_k10_TSCO_s105
                                 + 0.99 b_k10_TSCO_s105 - 1.01 s_k10_TSCO_s105
                                  = 0
 w_k10_TSCO_s106_return_constraint: 1.06599 w_k10_TSCO_s10 - w_k10_TSCO_s106
                                 + 0.99 b_k10_TSCO_s106 - 1.01 s_k10_TSCO_s106
                                  = 0
 w_k10_TSCO_s107_return_constraint: 1.07577 w_k10_TSCO_s10 - w_k10_TSCO_s107
                                 + 0.99 b_k10_TSCO_s107 - 1.01 s_k10_TSCO_s107
                                  = 0
 w_k10_TSCO_s110_return_constraint: 1.27357 w_k10_TSCO_s11 - w_k10_TSCO_s110
                                 + 0.99 b_k10_TSCO_s110 - 1.01 s_k10_TSCO_s110
                                  = 0
 w_k10_TSCO_s111_return_constraint: 0.85316 w_k10_TSCO_s11 - w_k10_TSCO_s111
                                 + 0.99 b_k10_TSCO_s111 - 1.01 s_k10_TSCO_s111
                                  = 0
 w_k10_TSCO_s112_return_constraint: 0.945394 w_k10_TSCO_s11 - w_k10_TSCO_s112
                                 + 0.99 b_k10_TSCO_s112 - 1.01 s_k10_TSCO_s112
                                  = 0
 w_k10_TSCO_s113_return_constraint: 0.957879 w_k10_TSCO_s11 - w_k10_TSCO_s113
                                 + 0.99 b_k10_TSCO_s113 - 1.01 s_k10_TSCO_s113
                                  = 0
 w_k10_TSCO_s114_return_constraint: 1.35504 w_k10_TSCO_s11 - w_k10_TSCO_s114
                                 + 0.99 b_k10_TSCO_s114 - 1.01 s_k10_TSCO_s114
                                  = 0
 w_k10_TSCO_s115_return_constraint: 0.727401 w_k10_TSCO_s11 - w_k10_TSCO_s115
                                 + 0.99 b_k10_TSCO_s115 - 1.01 s_k10_TSCO_s115
                                  = 0
 w_k10_TSCO_s116_return_constraint: 1.04477 w_k10_TSCO_s11 - w_k10_TSCO_s116
                                 + 0.99 b_k10_TSCO_s116 - 1.01 s_k10_TSCO_s116
                                  = 0
 w_k10_TSCO_s117_return_constraint: 1.05593 w_k10_TSCO_s11 - w_k10_TSCO_s117
                                 + 0.99 b_k10_TSCO_s117 - 1.01 s_k10_TSCO_s117
                                  = 0
 w_k10_AAL_s0_return_constraint: 1.00346 w_0_AAL - w_k10_AAL_s0
                                 + 0.99 b_k10_AAL_s0 - 1.01 s_k10_AAL_s0  = 0
 w_k10_AAL_s1_return_constraint: 1.01909 w_0_AAL - w_k10_AAL_s1
                                 + 0.99 b_k10_AAL_s1 - 1.01 s_k10_AAL_s1  = 0
 w_k10_AAL_s00_return_constraint: 1.08932 w_k10_AAL_s0 - w_k10_AAL_s00
                                 + 0.99 b_k10_AAL_s00 - 1.01 s_k10_AAL_s00  = 0
 w_k10_AAL_s01_return_constraint: 0.927571 w_k10_AAL_s0 - w_k10_AAL_s01
                                 + 0.99 b_k10_AAL_s01 - 1.01 s_k10_AAL_s01  = 0
 w_k10_AAL_s10_return_constraint: 0.995263 w_k10_AAL_s1 - w_k10_AAL_s10
                                 + 0.99 b_k10_AAL_s10 - 1.01 s_k10_AAL_s10  = 0
 w_k10_AAL_s11_return_constraint: 1.03947 w_k10_AAL_s1 - w_k10_AAL_s11
                                 + 0.99 b_k10_AAL_s11 - 1.01 s_k10_AAL_s11  = 0
 w_k10_AAL_s000_return_constraint: 1.01572 w_k10_AAL_s00 - w_k10_AAL_s000
                                 + 0.99 b_k10_AAL_s000 - 1.01 s_k10_AAL_s000
                                  = 0
 w_k10_AAL_s001_return_constraint: 0.966386 w_k10_AAL_s00 - w_k10_AAL_s001
                                 + 0.99 b_k10_AAL_s001 - 1.01 s_k10_AAL_s001
                                  = 0
 w_k10_AAL_s002_return_constraint: 0.939801 w_k10_AAL_s00 - w_k10_AAL_s002
                                 + 0.99 b_k10_AAL_s002 - 1.01 s_k10_AAL_s002
                                  = 0
 w_k10_AAL_s003_return_constraint: 1.06838 w_k10_AAL_s00 - w_k10_AAL_s003
                                 + 0.99 b_k10_AAL_s003 - 1.01 s_k10_AAL_s003
                                  = 0
 w_k10_AAL_s004_return_constraint: 0.877592 w_k10_AAL_s00 - w_k10_AAL_s004
                                 + 0.99 b_k10_AAL_s004 - 1.01 s_k10_AAL_s004
                                  = 0
 w_k10_AAL_s005_return_constraint: 1.18729 w_k10_AAL_s00 - w_k10_AAL_s005
                                 + 0.99 b_k10_AAL_s005 - 1.01 s_k10_AAL_s005
                                  = 0
 w_k10_AAL_s006_return_constraint: 1.1821 w_k10_AAL_s00 - w_k10_AAL_s006
                                 + 0.99 b_k10_AAL_s006 - 1.01 s_k10_AAL_s006
                                  = 0
 w_k10_AAL_s007_return_constraint: 1.01328 w_k10_AAL_s00 - w_k10_AAL_s007
                                 + 0.99 b_k10_AAL_s007 - 1.01 s_k10_AAL_s007
                                  = 0
 w_k10_AAL_s010_return_constraint: 0.867237 w_k10_AAL_s01 - w_k10_AAL_s010
                                 + 0.99 b_k10_AAL_s010 - 1.01 s_k10_AAL_s010
                                  = 0
 w_k10_AAL_s011_return_constraint: 0.952768 w_k10_AAL_s01 - w_k10_AAL_s011
                                 + 0.99 b_k10_AAL_s011 - 1.01 s_k10_AAL_s011
                                  = 0
 w_k10_AAL_s012_return_constraint: 0.966719 w_k10_AAL_s01 - w_k10_AAL_s012
                                 + 0.99 b_k10_AAL_s012 - 1.01 s_k10_AAL_s012
                                  = 0
 w_k10_AAL_s013_return_constraint: 1.01582 w_k10_AAL_s01 - w_k10_AAL_s013
                                 + 0.99 b_k10_AAL_s013 - 1.01 s_k10_AAL_s013
                                  = 0
 w_k10_AAL_s014_return_constraint: 0.951549 w_k10_AAL_s01 - w_k10_AAL_s014
                                 + 0.99 b_k10_AAL_s014 - 1.01 s_k10_AAL_s014
                                  = 0
 w_k10_AAL_s015_return_constraint: 1.1732 w_k10_AAL_s01 - w_k10_AAL_s015
                                 + 0.99 b_k10_AAL_s015 - 1.01 s_k10_AAL_s015
                                  = 0
 w_k10_AAL_s016_return_constraint: 0.950694 w_k10_AAL_s01 - w_k10_AAL_s016
                                 + 0.99 b_k10_AAL_s016 - 1.01 s_k10_AAL_s016
                                  = 0
 w_k10_AAL_s017_return_constraint: 1.16199 w_k10_AAL_s01 - w_k10_AAL_s017
                                 + 0.99 b_k10_AAL_s017 - 1.01 s_k10_AAL_s017
                                  = 0
 w_k10_AAL_s100_return_constraint: 0.910036 w_k10_AAL_s10 - w_k10_AAL_s100
                                 + 0.99 b_k10_AAL_s100 - 1.01 s_k10_AAL_s100
                                  = 0
 w_k10_AAL_s101_return_constraint: 0.797935 w_k10_AAL_s10 - w_k10_AAL_s101
                                 + 0.99 b_k10_AAL_s101 - 1.01 s_k10_AAL_s101
                                  = 0
 w_k10_AAL_s102_return_constraint: 1.07101 w_k10_AAL_s10 - w_k10_AAL_s102
                                 + 0.99 b_k10_AAL_s102 - 1.01 s_k10_AAL_s102
                                  = 0
 w_k10_AAL_s103_return_constraint: 1.12057 w_k10_AAL_s10 - w_k10_AAL_s103
                                 + 0.99 b_k10_AAL_s103 - 1.01 s_k10_AAL_s103
                                  = 0
 w_k10_AAL_s104_return_constraint: 0.903007 w_k10_AAL_s10 - w_k10_AAL_s104
                                 + 0.99 b_k10_AAL_s104 - 1.01 s_k10_AAL_s104
                                  = 0
 w_k10_AAL_s105_return_constraint: 0.879888 w_k10_AAL_s10 - w_k10_AAL_s105
                                 + 0.99 b_k10_AAL_s105 - 1.01 s_k10_AAL_s105
                                  = 0
 w_k10_AAL_s106_return_constraint: 1.18421 w_k10_AAL_s10 - w_k10_AAL_s106
                                 + 0.99 b_k10_AAL_s106 - 1.01 s_k10_AAL_s106
                                  = 0
 w_k10_AAL_s107_return_constraint: 1.2342 w_k10_AAL_s10 - w_k10_AAL_s107
                                 + 0.99 b_k10_AAL_s107 - 1.01 s_k10_AAL_s107
                                  = 0
 w_k10_AAL_s110_return_constraint: 1.13519 w_k10_AAL_s11 - w_k10_AAL_s110
                                 + 0.99 b_k10_AAL_s110 - 1.01 s_k10_AAL_s110
                                  = 0
 w_k10_AAL_s111_return_constraint: 0.972811 w_k10_AAL_s11 - w_k10_AAL_s111
                                 + 0.99 b_k10_AAL_s111 - 1.01 s_k10_AAL_s111
                                  = 0
 w_k10_AAL_s112_return_constraint: 0.952695 w_k10_AAL_s11 - w_k10_AAL_s112
                                 + 0.99 b_k10_AAL_s112 - 1.01 s_k10_AAL_s112
                                  = 0
 w_k10_AAL_s113_return_constraint: 0.852007 w_k10_AAL_s11 - w_k10_AAL_s113
                                 + 0.99 b_k10_AAL_s113 - 1.01 s_k10_AAL_s113
                                  = 0
 w_k10_AAL_s114_return_constraint: 0.819918 w_k10_AAL_s11 - w_k10_AAL_s114
                                 + 0.99 b_k10_AAL_s114 - 1.01 s_k10_AAL_s114
                                  = 0
 w_k10_AAL_s115_return_constraint: 1.19899 w_k10_AAL_s11 - w_k10_AAL_s115
                                 + 0.99 b_k10_AAL_s115 - 1.01 s_k10_AAL_s115
                                  = 0
 w_k10_AAL_s116_return_constraint: 0.951786 w_k10_AAL_s11 - w_k10_AAL_s116
                                 + 0.99 b_k10_AAL_s116 - 1.01 s_k10_AAL_s116
                                  = 0
 w_k10_AAL_s117_return_constraint: 1.18271 w_k10_AAL_s11 - w_k10_AAL_s117
                                 + 0.99 b_k10_AAL_s117 - 1.01 s_k10_AAL_s117
                                  = 0
 w_k10_PRU_s0_return_constraint: 1.03884 w_0_PRU - w_k10_PRU_s0
                                 + 0.99 b_k10_PRU_s0 - 1.01 s_k10_PRU_s0  = 0
 w_k10_PRU_s1_return_constraint: 1.02359 w_0_PRU - w_k10_PRU_s1
                                 + 0.99 b_k10_PRU_s1 - 1.01 s_k10_PRU_s1  = 0
 w_k10_PRU_s00_return_constraint: 1.15825 w_k10_PRU_s0 - w_k10_PRU_s00
                                 + 0.99 b_k10_PRU_s00 - 1.01 s_k10_PRU_s00  = 0
 w_k10_PRU_s01_return_constraint: 1.01351 w_k10_PRU_s0 - w_k10_PRU_s01
                                 + 0.99 b_k10_PRU_s01 - 1.01 s_k10_PRU_s01  = 0
 w_k10_PRU_s10_return_constraint: 0.939024 w_k10_PRU_s1 - w_k10_PRU_s10
                                 + 0.99 b_k10_PRU_s10 - 1.01 s_k10_PRU_s10  = 0
 w_k10_PRU_s11_return_constraint: 1.02056 w_k10_PRU_s1 - w_k10_PRU_s11
                                 + 0.99 b_k10_PRU_s11 - 1.01 s_k10_PRU_s11  = 0
 w_k10_PRU_s000_return_constraint: 1.18592 w_k10_PRU_s00 - w_k10_PRU_s000
                                 + 0.99 b_k10_PRU_s000 - 1.01 s_k10_PRU_s000
                                  = 0
 w_k10_PRU_s001_return_constraint: 0.78705 w_k10_PRU_s00 - w_k10_PRU_s001
                                 + 0.99 b_k10_PRU_s001 - 1.01 s_k10_PRU_s001
                                  = 0
 w_k10_PRU_s002_return_constraint: 0.987368 w_k10_PRU_s00 - w_k10_PRU_s002
                                 + 0.99 b_k10_PRU_s002 - 1.01 s_k10_PRU_s002
                                  = 0
 w_k10_PRU_s003_return_constraint: 1.0272 w_k10_PRU_s00 - w_k10_PRU_s003
                                 + 0.99 b_k10_PRU_s003 - 1.01 s_k10_PRU_s003
                                  = 0
 w_k10_PRU_s004_return_constraint: 0.906653 w_k10_PRU_s00 - w_k10_PRU_s004
                                 + 0.99 b_k10_PRU_s004 - 1.01 s_k10_PRU_s004
                                  = 0
 w_k10_PRU_s005_return_constraint: 1.09294 w_k10_PRU_s00 - w_k10_PRU_s005
                                 + 0.99 b_k10_PRU_s005 - 1.01 s_k10_PRU_s005
                                  = 0
 w_k10_PRU_s006_return_constraint: 0.883246 w_k10_PRU_s00 - w_k10_PRU_s006
                                 + 0.99 b_k10_PRU_s006 - 1.01 s_k10_PRU_s006
                                  = 0
 w_k10_PRU_s007_return_constraint: 1.10877 w_k10_PRU_s00 - w_k10_PRU_s007
                                 + 0.99 b_k10_PRU_s007 - 1.01 s_k10_PRU_s007
                                  = 0
 w_k10_PRU_s010_return_constraint: 0.928296 w_k10_PRU_s01 - w_k10_PRU_s010
                                 + 0.99 b_k10_PRU_s010 - 1.01 s_k10_PRU_s010
                                  = 0
 w_k10_PRU_s011_return_constraint: 1.01127 w_k10_PRU_s01 - w_k10_PRU_s011
                                 + 0.99 b_k10_PRU_s011 - 1.01 s_k10_PRU_s011
                                  = 0
 w_k10_PRU_s012_return_constraint: 0.908941 w_k10_PRU_s01 - w_k10_PRU_s012
                                 + 0.99 b_k10_PRU_s012 - 1.01 s_k10_PRU_s012
                                  = 0
 w_k10_PRU_s013_return_constraint: 0.944689 w_k10_PRU_s01 - w_k10_PRU_s013
                                 + 0.99 b_k10_PRU_s013 - 1.01 s_k10_PRU_s013
                                  = 0
 w_k10_PRU_s014_return_constraint: 0.875969 w_k10_PRU_s01 - w_k10_PRU_s014
                                 + 0.99 b_k10_PRU_s014 - 1.01 s_k10_PRU_s014
                                  = 0
 w_k10_PRU_s015_return_constraint: 1.06396 w_k10_PRU_s01 - w_k10_PRU_s015
                                 + 0.99 b_k10_PRU_s015 - 1.01 s_k10_PRU_s015
                                  = 0
 w_k10_PRU_s016_return_constraint: 0.744183 w_k10_PRU_s01 - w_k10_PRU_s016
                                 + 0.99 b_k10_PRU_s016 - 1.01 s_k10_PRU_s016
                                  = 0
 w_k10_PRU_s017_return_constraint: 1.22379 w_k10_PRU_s01 - w_k10_PRU_s017
                                 + 0.99 b_k10_PRU_s017 - 1.01 s_k10_PRU_s017
                                  = 0
 w_k10_PRU_s100_return_constraint: 1.23018 w_k10_PRU_s10 - w_k10_PRU_s100
                                 + 0.99 b_k10_PRU_s100 - 1.01 s_k10_PRU_s100
                                  = 0
 w_k10_PRU_s101_return_constraint: 0.667251 w_k10_PRU_s10 - w_k10_PRU_s101
                                 + 0.99 b_k10_PRU_s101 - 1.01 s_k10_PRU_s101
                                  = 0
 w_k10_PRU_s102_return_constraint: 0.969957 w_k10_PRU_s10 - w_k10_PRU_s102
                                 + 0.99 b_k10_PRU_s102 - 1.01 s_k10_PRU_s102
                                  = 0
 w_k10_PRU_s103_return_constraint: 1.14616 w_k10_PRU_s10 - w_k10_PRU_s103
                                 + 0.99 b_k10_PRU_s103 - 1.01 s_k10_PRU_s103
                                  = 0
 w_k10_PRU_s104_return_constraint: 0.890321 w_k10_PRU_s10 - w_k10_PRU_s104
                                 + 0.99 b_k10_PRU_s104 - 1.01 s_k10_PRU_s104
                                  = 0
 w_k10_PRU_s105_return_constraint: 0.874855 w_k10_PRU_s10 - w_k10_PRU_s105
                                 + 0.99 b_k10_PRU_s105 - 1.01 s_k10_PRU_s105
                                  = 0
 w_k10_PRU_s106_return_constraint: 1.21623 w_k10_PRU_s10 - w_k10_PRU_s106
                                 + 0.99 b_k10_PRU_s106 - 1.01 s_k10_PRU_s106
                                  = 0
 w_k10_PRU_s107_return_constraint: 1.16322 w_k10_PRU_s10 - w_k10_PRU_s107
                                 + 0.99 b_k10_PRU_s107 - 1.01 s_k10_PRU_s107
                                  = 0
 w_k10_PRU_s110_return_constraint: 1.39791 w_k10_PRU_s11 - w_k10_PRU_s110
                                 + 0.99 b_k10_PRU_s110 - 1.01 s_k10_PRU_s110
                                  = 0
 w_k10_PRU_s111_return_constraint: 1.04853 w_k10_PRU_s11 - w_k10_PRU_s111
                                 + 0.99 b_k10_PRU_s111 - 1.01 s_k10_PRU_s111
                                  = 0
 w_k10_PRU_s112_return_constraint: 0.73689 w_k10_PRU_s11 - w_k10_PRU_s112
                                 + 0.99 b_k10_PRU_s112 - 1.01 s_k10_PRU_s112
                                  = 0
 w_k10_PRU_s113_return_constraint: 0.986455 w_k10_PRU_s11 - w_k10_PRU_s113
                                 + 0.99 b_k10_PRU_s113 - 1.01 s_k10_PRU_s113
                                  = 0
 w_k10_PRU_s114_return_constraint: 1.0607 w_k10_PRU_s11 - w_k10_PRU_s114
                                 + 0.99 b_k10_PRU_s114 - 1.01 s_k10_PRU_s114
                                  = 0
 w_k10_PRU_s115_return_constraint: 1.07417 w_k10_PRU_s11 - w_k10_PRU_s115
                                 + 0.99 b_k10_PRU_s115 - 1.01 s_k10_PRU_s115
                                  = 0
 w_k10_PRU_s116_return_constraint: 1.11367 w_k10_PRU_s11 - w_k10_PRU_s116
                                 + 0.99 b_k10_PRU_s116 - 1.01 s_k10_PRU_s116
                                  = 0
 w_k10_PRU_s117_return_constraint: 1.20558 w_k10_PRU_s11 - w_k10_PRU_s117
                                 + 0.99 b_k10_PRU_s117 - 1.01 s_k10_PRU_s117
                                  = 0
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
 k10_s000_balance_constraint:    b_k10_HSBC_s000 + b_k10_VOD_s000
                                 + b_k10_BP_s000 + b_k10_GSK_s000
                                 + b_k10_AZN_s000 + b_k10_RIO_s000
                                 + b_k10_BG_s000 + b_k10_TSCO_s000
                                 + b_k10_AAL_s000 + b_k10_PRU_s000
                                 - s_k10_HSBC_s000 - s_k10_VOD_s000
                                 - s_k10_BP_s000 - s_k10_GSK_s000
                                 - s_k10_AZN_s000 - s_k10_RIO_s000
                                 - s_k10_BG_s000 - s_k10_TSCO_s000
                                 - s_k10_AAL_s000 - s_k10_PRU_s000  = 0
 k10_s001_balance_constraint:    b_k10_HSBC_s001 + b_k10_VOD_s001
                                 + b_k10_BP_s001 + b_k10_GSK_s001
                                 + b_k10_AZN_s001 + b_k10_RIO_s001
                                 + b_k10_BG_s001 + b_k10_TSCO_s001
                                 + b_k10_AAL_s001 + b_k10_PRU_s001
                                 - s_k10_HSBC_s001 - s_k10_VOD_s001
                                 - s_k10_BP_s001 - s_k10_GSK_s001
                                 - s_k10_AZN_s001 - s_k10_RIO_s001
                                 - s_k10_BG_s001 - s_k10_TSCO_s001
                                 - s_k10_AAL_s001 - s_k10_PRU_s001  = 0
 k10_s002_balance_constraint:    b_k10_HSBC_s002 + b_k10_VOD_s002
                                 + b_k10_BP_s002 + b_k10_GSK_s002
                                 + b_k10_AZN_s002 + b_k10_RIO_s002
                                 + b_k10_BG_s002 + b_k10_TSCO_s002
                                 + b_k10_AAL_s002 + b_k10_PRU_s002
                                 - s_k10_HSBC_s002 - s_k10_VOD_s002
                                 - s_k10_BP_s002 - s_k10_GSK_s002
                                 - s_k10_AZN_s002 - s_k10_RIO_s002
                                 - s_k10_BG_s002 - s_k10_TSCO_s002
                                 - s_k10_AAL_s002 - s_k10_PRU_s002  = 0
 k10_s003_balance_constraint:    b_k10_HSBC_s003 + b_k10_VOD_s003
                                 + b_k10_BP_s003 + b_k10_GSK_s003
                                 + b_k10_AZN_s003 + b_k10_RIO_s003
                                 + b_k10_BG_s003 + b_k10_TSCO_s003
                                 + b_k10_AAL_s003 + b_k10_PRU_s003
                                 - s_k10_HSBC_s003 - s_k10_VOD_s003
                                 - s_k10_BP_s003 - s_k10_GSK_s003
                                 - s_k10_AZN_s003 - s_k10_RIO_s003
                                 - s_k10_BG_s003 - s_k10_TSCO_s003
                                 - s_k10_AAL_s003 - s_k10_PRU_s003  = 0
 k10_s004_balance_constraint:    b_k10_HSBC_s004 + b_k10_VOD_s004
                                 + b_k10_BP_s004 + b_k10_GSK_s004
                                 + b_k10_AZN_s004 + b_k10_RIO_s004
                                 + b_k10_BG_s004 + b_k10_TSCO_s004
                                 + b_k10_AAL_s004 + b_k10_PRU_s004
                                 - s_k10_HSBC_s004 - s_k10_VOD_s004
                                 - s_k10_BP_s004 - s_k10_GSK_s004
                                 - s_k10_AZN_s004 - s_k10_RIO_s004
                                 - s_k10_BG_s004 - s_k10_TSCO_s004
                                 - s_k10_AAL_s004 - s_k10_PRU_s004  = 0
 k10_s005_balance_constraint:    b_k10_HSBC_s005 + b_k10_VOD_s005
                                 + b_k10_BP_s005 + b_k10_GSK_s005
                                 + b_k10_AZN_s005 + b_k10_RIO_s005
                                 + b_k10_BG_s005 + b_k10_TSCO_s005
                                 + b_k10_AAL_s005 + b_k10_PRU_s005
                                 - s_k10_HSBC_s005 - s_k10_VOD_s005
                                 - s_k10_BP_s005 - s_k10_GSK_s005
                                 - s_k10_AZN_s005 - s_k10_RIO_s005
                                 - s_k10_BG_s005 - s_k10_TSCO_s005
                                 - s_k10_AAL_s005 - s_k10_PRU_s005  = 0
 k10_s006_balance_constraint:    b_k10_HSBC_s006 + b_k10_VOD_s006
                                 + b_k10_BP_s006 + b_k10_GSK_s006
                                 + b_k10_AZN_s006 + b_k10_RIO_s006
                                 + b_k10_BG_s006 + b_k10_TSCO_s006
                                 + b_k10_AAL_s006 + b_k10_PRU_s006
                                 - s_k10_HSBC_s006 - s_k10_VOD_s006
                                 - s_k10_BP_s006 - s_k10_GSK_s006
                                 - s_k10_AZN_s006 - s_k10_RIO_s006
                                 - s_k10_BG_s006 - s_k10_TSCO_s006
                                 - s_k10_AAL_s006 - s_k10_PRU_s006  = 0
 k10_s007_balance_constraint:    b_k10_HSBC_s007 + b_k10_VOD_s007
                                 + b_k10_BP_s007 + b_k10_GSK_s007
                                 + b_k10_AZN_s007 + b_k10_RIO_s007
                                 + b_k10_BG_s007 + b_k10_TSCO_s007
                                 + b_k10_AAL_s007 + b_k10_PRU_s007
                                 - s_k10_HSBC_s007 - s_k10_VOD_s007
                                 - s_k10_BP_s007 - s_k10_GSK_s007
                                 - s_k10_AZN_s007 - s_k10_RIO_s007
                                 - s_k10_BG_s007 - s_k10_TSCO_s007
                                 - s_k10_AAL_s007 - s_k10_PRU_s007  = 0
 k10_s010_balance_constraint:    b_k10_HSBC_s010 + b_k10_VOD_s010
                                 + b_k10_BP_s010 + b_k10_GSK_s010
                                 + b_k10_AZN_s010 + b_k10_RIO_s010
                                 + b_k10_BG_s010 + b_k10_TSCO_s010
                                 + b_k10_AAL_s010 + b_k10_PRU_s010
                                 - s_k10_HSBC_s010 - s_k10_VOD_s010
                                 - s_k10_BP_s010 - s_k10_GSK_s010
                                 - s_k10_AZN_s010 - s_k10_RIO_s010
                                 - s_k10_BG_s010 - s_k10_TSCO_s010
                                 - s_k10_AAL_s010 - s_k10_PRU_s010  = 0
 k10_s011_balance_constraint:    b_k10_HSBC_s011 + b_k10_VOD_s011
                                 + b_k10_BP_s011 + b_k10_GSK_s011
                                 + b_k10_AZN_s011 + b_k10_RIO_s011
                                 + b_k10_BG_s011 + b_k10_TSCO_s011
                                 + b_k10_AAL_s011 + b_k10_PRU_s011
                                 - s_k10_HSBC_s011 - s_k10_VOD_s011
                                 - s_k10_BP_s011 - s_k10_GSK_s011
                                 - s_k10_AZN_s011 - s_k10_RIO_s011
                                 - s_k10_BG_s011 - s_k10_TSCO_s011
                                 - s_k10_AAL_s011 - s_k10_PRU_s011  = 0
 k10_s012_balance_constraint:    b_k10_HSBC_s012 + b_k10_VOD_s012
                                 + b_k10_BP_s012 + b_k10_GSK_s012
                                 + b_k10_AZN_s012 + b_k10_RIO_s012
                                 + b_k10_BG_s012 + b_k10_TSCO_s012
                                 + b_k10_AAL_s012 + b_k10_PRU_s012
                                 - s_k10_HSBC_s012 - s_k10_VOD_s012
                                 - s_k10_BP_s012 - s_k10_GSK_s012
                                 - s_k10_AZN_s012 - s_k10_RIO_s012
                                 - s_k10_BG_s012 - s_k10_TSCO_s012
                                 - s_k10_AAL_s012 - s_k10_PRU_s012  = 0
 k10_s013_balance_constraint:    b_k10_HSBC_s013 + b_k10_VOD_s013
                                 + b_k10_BP_s013 + b_k10_GSK_s013
                                 + b_k10_AZN_s013 + b_k10_RIO_s013
                                 + b_k10_BG_s013 + b_k10_TSCO_s013
                                 + b_k10_AAL_s013 + b_k10_PRU_s013
                                 - s_k10_HSBC_s013 - s_k10_VOD_s013
                                 - s_k10_BP_s013 - s_k10_GSK_s013
                                 - s_k10_AZN_s013 - s_k10_RIO_s013
                                 - s_k10_BG_s013 - s_k10_TSCO_s013
                                 - s_k10_AAL_s013 - s_k10_PRU_s013  = 0
 k10_s014_balance_constraint:    b_k10_HSBC_s014 + b_k10_VOD_s014
                                 + b_k10_BP_s014 + b_k10_GSK_s014
                                 + b_k10_AZN_s014 + b_k10_RIO_s014
                                 + b_k10_BG_s014 + b_k10_TSCO_s014
                                 + b_k10_AAL_s014 + b_k10_PRU_s014
                                 - s_k10_HSBC_s014 - s_k10_VOD_s014
                                 - s_k10_BP_s014 - s_k10_GSK_s014
                                 - s_k10_AZN_s014 - s_k10_RIO_s014
                                 - s_k10_BG_s014 - s_k10_TSCO_s014
                                 - s_k10_AAL_s014 - s_k10_PRU_s014  = 0
 k10_s015_balance_constraint:    b_k10_HSBC_s015 + b_k10_VOD_s015
                                 + b_k10_BP_s015 + b_k10_GSK_s015
                                 + b_k10_AZN_s015 + b_k10_RIO_s015
                                 + b_k10_BG_s015 + b_k10_TSCO_s015
                                 + b_k10_AAL_s015 + b_k10_PRU_s015
                                 - s_k10_HSBC_s015 - s_k10_VOD_s015
                                 - s_k10_BP_s015 - s_k10_GSK_s015
                                 - s_k10_AZN_s015 - s_k10_RIO_s015
                                 - s_k10_BG_s015 - s_k10_TSCO_s015
                                 - s_k10_AAL_s015 - s_k10_PRU_s015  = 0
 k10_s016_balance_constraint:    b_k10_HSBC_s016 + b_k10_VOD_s016
                                 + b_k10_BP_s016 + b_k10_GSK_s016
                                 + b_k10_AZN_s016 + b_k10_RIO_s016
                                 + b_k10_BG_s016 + b_k10_TSCO_s016
                                 + b_k10_AAL_s016 + b_k10_PRU_s016
                                 - s_k10_HSBC_s016 - s_k10_VOD_s016
                                 - s_k10_BP_s016 - s_k10_GSK_s016
                                 - s_k10_AZN_s016 - s_k10_RIO_s016
                                 - s_k10_BG_s016 - s_k10_TSCO_s016
                                 - s_k10_AAL_s016 - s_k10_PRU_s016  = 0
 k10_s017_balance_constraint:    b_k10_HSBC_s017 + b_k10_VOD_s017
                                 + b_k10_BP_s017 + b_k10_GSK_s017
                                 + b_k10_AZN_s017 + b_k10_RIO_s017
                                 + b_k10_BG_s017 + b_k10_TSCO_s017
                                 + b_k10_AAL_s017 + b_k10_PRU_s017
                                 - s_k10_HSBC_s017 - s_k10_VOD_s017
                                 - s_k10_BP_s017 - s_k10_GSK_s017
                                 - s_k10_AZN_s017 - s_k10_RIO_s017
                                 - s_k10_BG_s017 - s_k10_TSCO_s017
                                 - s_k10_AAL_s017 - s_k10_PRU_s017  = 0
 k10_s100_balance_constraint:    b_k10_HSBC_s100 + b_k10_VOD_s100
                                 + b_k10_BP_s100 + b_k10_GSK_s100
                                 + b_k10_AZN_s100 + b_k10_RIO_s100
                                 + b_k10_BG_s100 + b_k10_TSCO_s100
                                 + b_k10_AAL_s100 + b_k10_PRU_s100
                                 - s_k10_HSBC_s100 - s_k10_VOD_s100
                                 - s_k10_BP_s100 - s_k10_GSK_s100
                                 - s_k10_AZN_s100 - s_k10_RIO_s100
                                 - s_k10_BG_s100 - s_k10_TSCO_s100
                                 - s_k10_AAL_s100 - s_k10_PRU_s100  = 0
 k10_s101_balance_constraint:    b_k10_HSBC_s101 + b_k10_VOD_s101
                                 + b_k10_BP_s101 + b_k10_GSK_s101
                                 + b_k10_AZN_s101 + b_k10_RIO_s101
                                 + b_k10_BG_s101 + b_k10_TSCO_s101
                                 + b_k10_AAL_s101 + b_k10_PRU_s101
                                 - s_k10_HSBC_s101 - s_k10_VOD_s101
                                 - s_k10_BP_s101 - s_k10_GSK_s101
                                 - s_k10_AZN_s101 - s_k10_RIO_s101
                                 - s_k10_BG_s101 - s_k10_TSCO_s101
                                 - s_k10_AAL_s101 - s_k10_PRU_s101  = 0
 k10_s102_balance_constraint:    b_k10_HSBC_s102 + b_k10_VOD_s102
                                 + b_k10_BP_s102 + b_k10_GSK_s102
                                 + b_k10_AZN_s102 + b_k10_RIO_s102
                                 + b_k10_BG_s102 + b_k10_TSCO_s102
                                 + b_k10_AAL_s102 + b_k10_PRU_s102
                                 - s_k10_HSBC_s102 - s_k10_VOD_s102
                                 - s_k10_BP_s102 - s_k10_GSK_s102
                                 - s_k10_AZN_s102 - s_k10_RIO_s102
                                 - s_k10_BG_s102 - s_k10_TSCO_s102
                                 - s_k10_AAL_s102 - s_k10_PRU_s102  = 0
 k10_s103_balance_constraint:    b_k10_HSBC_s103 + b_k10_VOD_s103
                                 + b_k10_BP_s103 + b_k10_GSK_s103
                                 + b_k10_AZN_s103 + b_k10_RIO_s103
                                 + b_k10_BG_s103 + b_k10_TSCO_s103
                                 + b_k10_AAL_s103 + b_k10_PRU_s103
                                 - s_k10_HSBC_s103 - s_k10_VOD_s103
                                 - s_k10_BP_s103 - s_k10_GSK_s103
                                 - s_k10_AZN_s103 - s_k10_RIO_s103
                                 - s_k10_BG_s103 - s_k10_TSCO_s103
                                 - s_k10_AAL_s103 - s_k10_PRU_s103  = 0
 k10_s104_balance_constraint:    b_k10_HSBC_s104 + b_k10_VOD_s104
                                 + b_k10_BP_s104 + b_k10_GSK_s104
                                 + b_k10_AZN_s104 + b_k10_RIO_s104
                                 + b_k10_BG_s104 + b_k10_TSCO_s104
                                 + b_k10_AAL_s104 + b_k10_PRU_s104
                                 - s_k10_HSBC_s104 - s_k10_VOD_s104
                                 - s_k10_BP_s104 - s_k10_GSK_s104
                                 - s_k10_AZN_s104 - s_k10_RIO_s104
                                 - s_k10_BG_s104 - s_k10_TSCO_s104
                                 - s_k10_AAL_s104 - s_k10_PRU_s104  = 0
 k10_s105_balance_constraint:    b_k10_HSBC_s105 + b_k10_VOD_s105
                                 + b_k10_BP_s105 + b_k10_GSK_s105
                                 + b_k10_AZN_s105 + b_k10_RIO_s105
                                 + b_k10_BG_s105 + b_k10_TSCO_s105
                                 + b_k10_AAL_s105 + b_k10_PRU_s105
                                 - s_k10_HSBC_s105 - s_k10_VOD_s105
                                 - s_k10_BP_s105 - s_k10_GSK_s105
                                 - s_k10_AZN_s105 - s_k10_RIO_s105
                                 - s_k10_BG_s105 - s_k10_TSCO_s105
                                 - s_k10_AAL_s105 - s_k10_PRU_s105  = 0
 k10_s106_balance_constraint:    b_k10_HSBC_s106 + b_k10_VOD_s106
                                 + b_k10_BP_s106 + b_k10_GSK_s106
                                 + b_k10_AZN_s106 + b_k10_RIO_s106
                                 + b_k10_BG_s106 + b_k10_TSCO_s106
                                 + b_k10_AAL_s106 + b_k10_PRU_s106
                                 - s_k10_HSBC_s106 - s_k10_VOD_s106
                                 - s_k10_BP_s106 - s_k10_GSK_s106
                                 - s_k10_AZN_s106 - s_k10_RIO_s106
                                 - s_k10_BG_s106 - s_k10_TSCO_s106
                                 - s_k10_AAL_s106 - s_k10_PRU_s106  = 0
 k10_s107_balance_constraint:    b_k10_HSBC_s107 + b_k10_VOD_s107
                                 + b_k10_BP_s107 + b_k10_GSK_s107
                                 + b_k10_AZN_s107 + b_k10_RIO_s107
                                 + b_k10_BG_s107 + b_k10_TSCO_s107
                                 + b_k10_AAL_s107 + b_k10_PRU_s107
                                 - s_k10_HSBC_s107 - s_k10_VOD_s107
                                 - s_k10_BP_s107 - s_k10_GSK_s107
                                 - s_k10_AZN_s107 - s_k10_RIO_s107
                                 - s_k10_BG_s107 - s_k10_TSCO_s107
                                 - s_k10_AAL_s107 - s_k10_PRU_s107  = 0
 k10_s110_balance_constraint:    b_k10_HSBC_s110 + b_k10_VOD_s110
                                 + b_k10_BP_s110 + b_k10_GSK_s110
                                 + b_k10_AZN_s110 + b_k10_RIO_s110
                                 + b_k10_BG_s110 + b_k10_TSCO_s110
                                 + b_k10_AAL_s110 + b_k10_PRU_s110
                                 - s_k10_HSBC_s110 - s_k10_VOD_s110
                                 - s_k10_BP_s110 - s_k10_GSK_s110
                                 - s_k10_AZN_s110 - s_k10_RIO_s110
                                 - s_k10_BG_s110 - s_k10_TSCO_s110
                                 - s_k10_AAL_s110 - s_k10_PRU_s110  = 0
 k10_s111_balance_constraint:    b_k10_HSBC_s111 + b_k10_VOD_s111
                                 + b_k10_BP_s111 + b_k10_GSK_s111
                                 + b_k10_AZN_s111 + b_k10_RIO_s111
                                 + b_k10_BG_s111 + b_k10_TSCO_s111
                                 + b_k10_AAL_s111 + b_k10_PRU_s111
                                 - s_k10_HSBC_s111 - s_k10_VOD_s111
                                 - s_k10_BP_s111 - s_k10_GSK_s111
                                 - s_k10_AZN_s111 - s_k10_RIO_s111
                                 - s_k10_BG_s111 - s_k10_TSCO_s111
                                 - s_k10_AAL_s111 - s_k10_PRU_s111  = 0
 k10_s112_balance_constraint:    b_k10_HSBC_s112 + b_k10_VOD_s112
                                 + b_k10_BP_s112 + b_k10_GSK_s112
                                 + b_k10_AZN_s112 + b_k10_RIO_s112
                                 + b_k10_BG_s112 + b_k10_TSCO_s112
                                 + b_k10_AAL_s112 + b_k10_PRU_s112
                                 - s_k10_HSBC_s112 - s_k10_VOD_s112
                                 - s_k10_BP_s112 - s_k10_GSK_s112
                                 - s_k10_AZN_s112 - s_k10_RIO_s112
                                 - s_k10_BG_s112 - s_k10_TSCO_s112
                                 - s_k10_AAL_s112 - s_k10_PRU_s112  = 0
 k10_s113_balance_constraint:    b_k10_HSBC_s113 + b_k10_VOD_s113
                                 + b_k10_BP_s113 + b_k10_GSK_s113
                                 + b_k10_AZN_s113 + b_k10_RIO_s113
                                 + b_k10_BG_s113 + b_k10_TSCO_s113
                                 + b_k10_AAL_s113 + b_k10_PRU_s113
                                 - s_k10_HSBC_s113 - s_k10_VOD_s113
                                 - s_k10_BP_s113 - s_k10_GSK_s113
                                 - s_k10_AZN_s113 - s_k10_RIO_s113
                                 - s_k10_BG_s113 - s_k10_TSCO_s113
                                 - s_k10_AAL_s113 - s_k10_PRU_s113  = 0
 k10_s114_balance_constraint:    b_k10_HSBC_s114 + b_k10_VOD_s114
                                 + b_k10_BP_s114 + b_k10_GSK_s114
                                 + b_k10_AZN_s114 + b_k10_RIO_s114
                                 + b_k10_BG_s114 + b_k10_TSCO_s114
                                 + b_k10_AAL_s114 + b_k10_PRU_s114
                                 - s_k10_HSBC_s114 - s_k10_VOD_s114
                                 - s_k10_BP_s114 - s_k10_GSK_s114
                                 - s_k10_AZN_s114 - s_k10_RIO_s114
                                 - s_k10_BG_s114 - s_k10_TSCO_s114
                                 - s_k10_AAL_s114 - s_k10_PRU_s114  = 0
 k10_s115_balance_constraint:    b_k10_HSBC_s115 + b_k10_VOD_s115
                                 + b_k10_BP_s115 + b_k10_GSK_s115
                                 + b_k10_AZN_s115 + b_k10_RIO_s115
                                 + b_k10_BG_s115 + b_k10_TSCO_s115
                                 + b_k10_AAL_s115 + b_k10_PRU_s115
                                 - s_k10_HSBC_s115 - s_k10_VOD_s115
                                 - s_k10_BP_s115 - s_k10_GSK_s115
                                 - s_k10_AZN_s115 - s_k10_RIO_s115
                                 - s_k10_BG_s115 - s_k10_TSCO_s115
                                 - s_k10_AAL_s115 - s_k10_PRU_s115  = 0
 k10_s116_balance_constraint:    b_k10_HSBC_s116 + b_k10_VOD_s116
                                 + b_k10_BP_s116 + b_k10_GSK_s116
                                 + b_k10_AZN_s116 + b_k10_RIO_s116
                                 + b_k10_BG_s116 + b_k10_TSCO_s116
                                 + b_k10_AAL_s116 + b_k10_PRU_s116
                                 - s_k10_HSBC_s116 - s_k10_VOD_s116
                                 - s_k10_BP_s116 - s_k10_GSK_s116
                                 - s_k10_AZN_s116 - s_k10_RIO_s116
                                 - s_k10_BG_s116 - s_k10_TSCO_s116
                                 - s_k10_AAL_s116 - s_k10_PRU_s116  = 0
 k10_s117_balance_constraint:    b_k10_HSBC_s117 + b_k10_VOD_s117
                                 + b_k10_BP_s117 + b_k10_GSK_s117
                                 + b_k10_AZN_s117 + b_k10_RIO_s117
                                 + b_k10_BG_s117 + b_k10_TSCO_s117
                                 + b_k10_AAL_s117 + b_k10_PRU_s117
                                 - s_k10_HSBC_s117 - s_k10_VOD_s117
                                 - s_k10_BP_s117 - s_k10_GSK_s117
                                 - s_k10_AZN_s117 - s_k10_RIO_s117
                                 - s_k10_BG_s117 - s_k10_TSCO_s117
                                 - s_k10_AAL_s117 - s_k10_PRU_s117  = 0
 k10_wcvar_constraint:           wcvar - var_10 - 0.167500381055984 z_k10_s0000
                                 - 0.134999704508011 z_k10_s0001
                                 - 0.0799999480417427 z_k10_s0002
                                 - 0.120000397062454 z_k10_s0003
                                 - 0.314999706351892 z_k10_s0004
                                 - 0.120000397062454 z_k10_s0005
                                 - 0.0975000108945989 z_k10_s0006
                                 - 0.152499886110827 z_k10_s0007
                                 - 0.174999244529888 z_k10_s0010
                                 - 0.182498760903868 z_k10_s0011
                                 - 0.187499546816733 z_k10_s0012
                                 - 0.0874997885144755 z_k10_s0013
                                 - 0.199999849103579 z_k10_s0014
                                 - 0.434998549615945 z_k10_s0015
                                 - 0.184999153860301 z_k10_s0016
                                 - 0.209999758433992 z_k10_s0017
                                 - 0.174999003570185 z_k10_s0020
                                 - 0.277498724517606 z_k10_s0021
                                 - 0.202500088699262 z_k10_s0022
                                 - 0.287499417291004 z_k10_s0023
                                 - 0.184999696343583 z_k10_s0024
                                 - 0.194998749121445 z_k10_s0025
                                 - 0.229999533832803 z_k10_s0026
                                 - 0.0874998297841993 z_k10_s0027
                                 - 0.254999944276671 z_k10_s0030
                                 - 0.142499850404677 z_k10_s0031
                                 - 0.16749987126512 z_k10_s0032
                                 - 0.639998923028253 z_k10_s0033
                                 - 0.169999873351164 z_k10_s0034
                                 - 0.54750072534355 z_k10_s0035
                                 - 0.395000866594696 z_k10_s0036
                                 - 0.367501112148057 z_k10_s0037
                                 - 0.422499423281364 z_k10_s0040
                                 - 0.462498655783647 z_k10_s0041
                                 - 0.224999196814142 z_k10_s0042
                                 - 0.329998688127835 z_k10_s0043
                                 - 0.397499902967437 z_k10_s0044
                                 - 0.222499244782749 z_k10_s0045
                                 - 0.162499392032525 z_k10_s0046
                                 - 0.287499503594159 z_k10_s0047
                                 - 0.182499211138045 z_k10_s0050
                                 - 0.147500300816054 z_k10_s0051
                                 - 0.129999825406141 z_k10_s0052
                                 - 0.33500018914207 z_k10_s0053
                                 - 0.149999910660813 z_k10_s0054
                                 - 0.317500005231849 z_k10_s0055
                                 - 0.0874998993021928 z_k10_s0056
                                 - 0.107499838807019 z_k10_s0057
                                 - 0.327499331074306 z_k10_s0060
                                 - 0.164999352742825 z_k10_s0061
                                 - 0.072499753610338 z_k10_s0062
                                 - 0.29249905785013 z_k10_s0063
                                 - 0.227499267216768 z_k10_s0064
                                 - 0.0774996731785357 z_k10_s0065
                                 - 0.359999226390987 z_k10_s0066
                                 - 0.149999426788874 z_k10_s0067
                                 - 0.382499001058953 z_k10_s0070
                                 - 0.127499476020359 z_k10_s0071
                                 - 0.122499496568581 z_k10_s0072
                                 - 0.204999539521514 z_k10_s0073
                                 - 0.257499514764483 z_k10_s0074
                                 - 0.327498845090803 z_k10_s0075
                                 - 0.369998097433048 z_k10_s0076
                                 - 0.117499517116802 z_k10_s0077
                                 - 0.11749999824926 z_k10_s0100
                                 - 0.269999328827966 z_k10_s0101
                                 - 0.120000036701379 z_k10_s0102
                                 - 0.160000048935172 z_k10_s0103
                                 - 0.11749999824926 z_k10_s0104
                                 - 0.362499545556146 z_k10_s0105
                                 - 0.485001027709898 z_k10_s0106
                                 - 0.377500178268932 z_k10_s0107
                                 - 0.107499991612463 z_k10_s0110
                                 - 0.157500058978762 z_k10_s0111
                                 - 0.225000813898609 z_k10_s0112
                                 - 0.167500038402004 z_k10_s0113
                                 - 0.250000336831494 z_k10_s0114
                                 - 0.145000127262231 z_k10_s0115
                                 - 0.339999640890409 z_k10_s0116
                                 - 0.30999987287077 z_k10_s0117
                                 - 1.44000128819809 z_k10_s0120
                                 - 0.322499777643211 z_k10_s0121
                                 - 0.30249986454326 z_k10_s0122
                                 - 1.3825020881921 z_k10_s0123
                                 - 1.10250078979224 z_k10_s0124
                                 - 0.647499465830149 z_k10_s0125
                                 - 0.435000153361872 z_k10_s0126
                                 - 0.655000454961603 z_k10_s0127
                                 - 0.209999912861958 z_k10_s0130
                                 - 0.232500375669075 z_k10_s0131
                                 - 0.140000107158139 z_k10_s0132
                                 - 0.217500617964888 z_k10_s0133
                                 - 0.132500063055879 z_k10_s0134
                                 - 0.0800000848403892 z_k10_s0135
                                 - 0.305000581657371 z_k10_s0136
                                 - 0.335000097065743 z_k10_s0137
                                 - 0.879997272341032 z_k10_s0140
                                 - 0.717499736212911 z_k10_s0141
                                 - 0.397499792702443 z_k10_s0142
                                 - 1.13500122765055 z_k10_s0143
                                 - 0.592499862306488 z_k10_s0144
                                 - 0.810000102893709 z_k10_s0145
                                 - 0.930003032304177 z_k10_s0146
                                 - 0.590000179558391 z_k10_s0147
                                 - 0.282500660860808 z_k10_s0150
                                 - 0.300000142728072 z_k10_s0151
                                 - 0.252499744087647 z_k10_s0152
                                 - 0.160000136288329 z_k10_s0153
                                 - 0.140000006439744 z_k10_s0154
                                 - 0.202499419466183 z_k10_s0155
                                 - 0.125000089553375 z_k10_s0156
                                 - 0.342500689406423 z_k10_s0157
                                 - 1.52249849357412 z_k10_s0160
                                 - 0.680000610331897 z_k10_s0161
                                 - 0.439999804803018 z_k10_s0162
                                 - 0.485000186839673 z_k10_s0163
                                 - 0.447500044475786 z_k10_s0164
                                 - 0.485000186839673 z_k10_s0165
                                 - 0.77499836618719 z_k10_s0166
                                 - 0.444999964584864 z_k10_s0167
                                 - 0.0950000454996869 z_k10_s0170
                                 - 0.150000098367948 z_k10_s0171
                                 - 0.210000171315155 z_k10_s0172
                                 - 0.407501132433632 z_k10_s0173
                                 - 0.0950000454996869 z_k10_s0174
                                 - 0.160000138525839 z_k10_s0175
                                 - 0.432499552826988 z_k10_s0176
                                 - 0.130000186052304 z_k10_s0177
                                 - 0.579998284465691 z_k10_s1000
                                 - 0.232499913716466 z_k10_s1001
                                 - 0.39999914195889 z_k10_s1002
                                 - 0.457500570223864 z_k10_s1003
                                 - 0.229999742313711 z_k10_s1004
                                 - 0.404998856264801 z_k10_s1005
                                 - 0.472499713141599 z_k10_s1006
                                 - 0.365001141817507 z_k10_s1007
                                 - 0.879995198941944 z_k10_s1010
                                 - 0.589996647188734 z_k10_s1011
                                 - 0.552498653004199 z_k10_s1012
                                 - 0.399998835493405 z_k10_s1013
                                 - 1.03249855194255 z_k10_s1014
                                 - 0.687499563695091 z_k10_s1015
                                 - 0.879995198941944 z_k10_s1016
                                 - 0.869995655259627 z_k10_s1017
                                 - 1.13749755751812 z_k10_s1020
                                 - 0.372499897670021 z_k10_s1021
                                 - 0.647499366124889 z_k10_s1022
                                 - 0.554999073536031 z_k10_s1023
                                 - 1.1825014691519 z_k10_s1024
                                 - 1.79499809089464 z_k10_s1025
                                 - 1.46499622555149 z_k10_s1026
                                 - 1.7849942416459 z_k10_s1027
                                 - 0.732499182759853 z_k10_s1030
                                 - 1.90499926838761 z_k10_s1031
                                 - 0.59749962761779 z_k10_s1032
                                 - 1.86749687112863 z_k10_s1033
                                 - 0.922495447639056 z_k10_s1034
                                 - 0.414999304695251 z_k10_s1035
                                 - 0.467499635861068 z_k10_s1036
                                 - 0.654999522168949 z_k10_s1037
                                 - 0.537499893469313 z_k10_s1040
                                 - 0.837497680615154 z_k10_s1041
                                 - 1.09000364469559 z_k10_s1042
                                 - 0.779999703128736 z_k10_s1043
                                 - 1.04750043071689 z_k10_s1044
                                 - 1.92499627301828 z_k10_s1045
                                 - 0.972497676930701 z_k10_s1046
                                 - 1.07750318573122 z_k10_s1047
                                 - 1.03250343318068 z_k10_s1050
                                 - 1.11999939683679 z_k10_s1051
                                 - 0.407500018862671 z_k10_s1052
                                 - 0.380000209819937 z_k10_s1053
                                 - 0.567500595839857 z_k10_s1054
                                 - 0.360000137697789 z_k10_s1055
                                 - 0.500000642552826 z_k10_s1056
                                 - 1.43499995251018 z_k10_s1057
                                 - 0.249999730409134 z_k10_s1060
                                 - 0.222499807814097 z_k10_s1061
                                 - 0.262500671928943 z_k10_s1062
                                 - 0.167499962624023 z_k10_s1063
                                 - 0.627499036827121 z_k10_s1064
                                 - 0.402498801959224 z_k10_s1065
                                 - 0.189999986110812 z_k10_s1066
                                 - 0.265000382733229 z_k10_s1067
                                 - 0.602501651829764 z_k10_s1070
                                 - 0.384999198099083 z_k10_s1071
                                 - 0.237500044476592 z_k10_s1072
                                 - 0.229999910072096 z_k10_s1073
                                 - 0.734999453643494 z_k10_s1074
                                 - 0.287499977840101 z_k10_s1075
                                 - 0.437499416930706 z_k10_s1076
                                 - 0.69500066215244 z_k10_s1077
                                 - 0.195000644482884 z_k10_s1100
                                 - 0.100000092926865 z_k10_s1101
                                 - 0.135000610026788 z_k10_s1102
                                 - 0.11499988854066 z_k10_s1103
                                 - 0.162500004568498 z_k10_s1104
                                 - 0.0650001083275137 z_k10_s1105
                                 - 0.0875000679984926 z_k10_s1106
                                 - 0.204999940224804 z_k10_s1107
                                 - 0.295000145451548 z_k10_s1110
                                 - 0.267499734689357 z_k10_s1111
                                 - 0.609999053731079 z_k10_s1112
                                 - 0.122499847941874 z_k10_s1113
                                 - 0.419998544372708 z_k10_s1114
                                 - 0.314998908279531 z_k10_s1115
                                 - 0.214999916642769 z_k10_s1116
                                 - 0.480000282853338 z_k10_s1117
                                 - 0.0999999329418651 z_k10_s1120
                                 - 0.0650001074124885 z_k10_s1121
                                 - 0.0525000780446863 z_k10_s1122
                                 - 0.0899999396476786 z_k10_s1123
                                 - 0.055000114118302 z_k10_s1124
                                 - 0.125000293678022 z_k10_s1125
                                 - 0.195000246737327 z_k10_s1126
                                 - 0.0725001401331975 z_k10_s1127
                                 - 0.0800001575253865 z_k10_s1130
                                 - 0.195000090858325 z_k10_s1131
                                 - 0.0825001586902368 z_k10_s1132
                                 - 0.0900001621847878 z_k10_s1133
                                 - 0.0925001633496381 z_k10_s1134
                                 - 0.352500164243895 z_k10_s1135
                                 - 0.117500174998141 z_k10_s1136
                                 - 0.192500089693475 z_k10_s1137
                                 - 0.134999971069398 z_k10_s1140
                                 - 0.107500161546209 z_k10_s1141
                                 - 0.0800001305227714 z_k10_s1142
                                 - 0.137499601366566 z_k10_s1143
                                 - 0.212500692795249 z_k10_s1144
                                 - 0.14500070726055 z_k10_s1145
                                 - 0.122499604581078 z_k10_s1146
                                 - 0.167500702438783 z_k10_s1147
                                 - 0.105000102762491 z_k10_s1150
                                 - 0.137500027069759 z_k10_s1151
                                 - 0.197500601791998 z_k10_s1152
                                 - 0.275000054139519 z_k10_s1153
                                 - 0.147500122856799 z_k10_s1154
                                 - 0.125000165336367 z_k10_s1155
                                 - 0.112500174602771 z_k10_s1156
                                 - 0.190000529951718 z_k10_s1157
                                 - 0.415000177802333 z_k10_s1160
                                 - 0.21000051599757 z_k10_s1161
                                 - 0.215000590188164 z_k10_s1162
                                 - 0.410000363612477 z_k10_s1163
                                 - 0.52000147580407 z_k10_s1164
                                 - 0.195000553426526 z_k10_s1165
                                 - 0.377500271374722 z_k10_s1166
                                 - 0.257500830807107 z_k10_s1167
                                 - 0.412499792140783 z_k10_s1170
                                 - 0.280000707000385 z_k10_s1171
                                 - 0.317500911286383 z_k10_s1172
                                 - 0.235000461857188 z_k10_s1173
                                 - 0.257500584428787 z_k10_s1174
                                 - 0.460001687573174 z_k10_s1175
                                 - 0.387500474285181 z_k10_s1176
                                 - 0.105000326500274 z_k10_s1177 >= 0
 z_k10_s0000_constraint:         var_10 + z_k10_s0000 + 1.07119 w_k10_HSBC_s000
                                 + 1.25837 w_k10_VOD_s000
                                 + 1.30279 w_k10_BP_s000
                                 + 1.1953 w_k10_GSK_s000
                                 + 1.13563 w_k10_AZN_s000
                                 + 1.261 w_k10_RIO_s000 + 1.08995 w_k10_BG_s000
                                 + 1.32532 w_k10_TSCO_s000
                                 + 1.17047 w_k10_AAL_s000
                                 + 1.16008 w_k10_PRU_s000 >= 1
 z_k10_s0001_constraint:         var_10 + z_k10_s0001
                                 + 0.874936 w_k10_HSBC_s000
                                 + 1.0477 w_k10_VOD_s000
                                 + 0.743773 w_k10_BP_s000
                                 + 0.749908 w_k10_GSK_s000
                                 + 0.960783 w_k10_AZN_s000
                                 + 0.962752 w_k10_RIO_s000
                                 + 0.948719 w_k10_BG_s000
                                 + 0.700024 w_k10_TSCO_s000
                                 + 0.890681 w_k10_AAL_s000
                                 + 0.869617 w_k10_PRU_s000 >= 1
 z_k10_s0002_constraint:         var_10 + z_k10_s0002
                                 + 0.667981 w_k10_HSBC_s000
                                 + 1.31589 w_k10_VOD_s000
                                 + 0.935666 w_k10_BP_s000
                                 + 0.941948 w_k10_GSK_s000
                                 + 1.11244 w_k10_AZN_s000
                                 + 0.893722 w_k10_RIO_s000
                                 + 0.858382 w_k10_BG_s000
                                 + 0.761708 w_k10_TSCO_s000
                                 + 1.20062 w_k10_AAL_s000
                                 + 1.1875 w_k10_PRU_s000 >= 1
 z_k10_s0003_constraint:         var_10 + z_k10_s0003 + 1.19754 w_k10_HSBC_s000
                                 + 1.21189 w_k10_VOD_s000
                                 + 0.979994 w_k10_BP_s000
                                 + 0.939936 w_k10_GSK_s000
                                 + 0.946672 w_k10_AZN_s000
                                 + 1.1599 w_k10_RIO_s000
                                 + 1.07297 w_k10_BG_s000
                                 + 1.02036 w_k10_TSCO_s000
                                 + 0.834027 w_k10_AAL_s000
                                 + 0.991442 w_k10_PRU_s000 >= 1
 z_k10_s0004_constraint:         var_10 + z_k10_s0004 + 1.10024 w_k10_HSBC_s000
                                 + 1.36439 w_k10_VOD_s000
                                 + 1.08556 w_k10_BP_s000
                                 + 1.02496 w_k10_GSK_s000
                                 + 1.17684 w_k10_AZN_s000
                                 + 1.15947 w_k10_RIO_s000
                                 + 1.0107 w_k10_BG_s000
                                 + 1.13937 w_k10_TSCO_s000
                                 + 1.00321 w_k10_AAL_s000
                                 + 1.19253 w_k10_PRU_s000 >= 1
 z_k10_s0005_constraint:         var_10 + z_k10_s0005
                                 + 0.863868 w_k10_HSBC_s000
                                 + 1.24199 w_k10_VOD_s000
                                 + 0.992408 w_k10_BP_s000
                                 + 0.912977 w_k10_GSK_s000
                                 + 1.08775 w_k10_AZN_s000
                                 + 1.09712 w_k10_RIO_s000
                                 + 1.07701 w_k10_BG_s000
                                 + 0.866737 w_k10_TSCO_s000
                                 + 1.12145 w_k10_AAL_s000
                                 + 1.2522 w_k10_PRU_s000 >= 1
 z_k10_s0006_constraint:         var_10 + z_k10_s0006
                                 + 0.803657 w_k10_HSBC_s000
                                 + 1.11513 w_k10_VOD_s000
                                 + 0.77114 w_k10_BP_s000
                                 + 0.936162 w_k10_GSK_s000
                                 + 1.04637 w_k10_AZN_s000
                                 + 0.853213 w_k10_RIO_s000
                                 + 0.879934 w_k10_BG_s000
                                 + 0.907633 w_k10_TSCO_s000
                                 + 1.1902 w_k10_AAL_s000
                                 + 1.02392 w_k10_PRU_s000 >= 1
 z_k10_s0007_constraint:         var_10 + z_k10_s0007
                                 + 0.865009 w_k10_HSBC_s000
                                 + 1.22284 w_k10_VOD_s000
                                 + 0.836735 w_k10_BP_s000
                                 + 0.874086 w_k10_GSK_s000
                                 + 1.03564 w_k10_AZN_s000
                                 + 0.988467 w_k10_RIO_s000
                                 + 1.09052 w_k10_BG_s000
                                 + 0.903033 w_k10_TSCO_s000
                                 + 0.949879 w_k10_AAL_s000
                                 + 1.04501 w_k10_PRU_s000 >= 1
 z_k10_s0010_constraint:         var_10 + z_k10_s0010
                                 + 0.993348 w_k10_HSBC_s001
                                 + 1.27499 w_k10_VOD_s001
                                 + 1.34036 w_k10_BP_s001
                                 + 1.1416 w_k10_GSK_s001
                                 + 1.10135 w_k10_AZN_s001
                                 + 1.32855 w_k10_RIO_s001
                                 + 1.00925 w_k10_BG_s001
                                 + 1.24426 w_k10_TSCO_s001
                                 + 1.07362 w_k10_AAL_s001
                                 + 1.41556 w_k10_PRU_s001 >= 1
 z_k10_s0011_constraint:         var_10 + z_k10_s0011 + 1.10528 w_k10_HSBC_s001
                                 + 1.1712 w_k10_VOD_s001
                                 + 1.10095 w_k10_BP_s001
                                 + 1.07345 w_k10_GSK_s001
                                 + 1.08002 w_k10_AZN_s001
                                 + 1.14544 w_k10_RIO_s001
                                 + 1.12614 w_k10_BG_s001
                                 + 1.18491 w_k10_TSCO_s001
                                 + 0.982846 w_k10_AAL_s001
                                 + 0.899577 w_k10_PRU_s001 >= 1
 z_k10_s0012_constraint:         var_10 + z_k10_s0012
                                 + 0.782212 w_k10_HSBC_s001
                                 + 1.08518 w_k10_VOD_s001
                                 + 0.837213 w_k10_BP_s001
                                 + 0.933616 w_k10_GSK_s001
                                 + 1.01215 w_k10_AZN_s001
                                 + 0.879377 w_k10_RIO_s001
                                 + 0.945258 w_k10_BG_s001
                                 + 0.814372 w_k10_TSCO_s001
                                 + 1.1043 w_k10_AAL_s001
                                 + 1.15117 w_k10_PRU_s001 >= 1
 z_k10_s0013_constraint:         var_10 + z_k10_s0013 + 1.30761 w_k10_HSBC_s001
                                 + 1.16239 w_k10_VOD_s001
                                 + 1.28423 w_k10_BP_s001
                                 + 1.04899 w_k10_GSK_s001
                                 + 1.03689 w_k10_AZN_s001
                                 + 1.23709 w_k10_RIO_s001
                                 + 0.998957 w_k10_BG_s001
                                 + 1.28595 w_k10_TSCO_s001
                                 + 0.910553 w_k10_AAL_s001
                                 + 1.48476 w_k10_PRU_s001 >= 1
 z_k10_s0014_constraint:         var_10 + z_k10_s0014
                                 + 0.894689 w_k10_HSBC_s001
                                 + 0.993464 w_k10_VOD_s001
                                 + 1.00322 w_k10_BP_s001
                                 + 0.920096 w_k10_GSK_s001
                                 + 1.0052 w_k10_AZN_s001
                                 + 1.05358 w_k10_RIO_s001
                                 + 1.06994 w_k10_BG_s001
                                 + 0.938422 w_k10_TSCO_s001
                                 + 0.859275 w_k10_AAL_s001
                                 + 1.0172 w_k10_PRU_s001 >= 1
 z_k10_s0015_constraint:         var_10 + z_k10_s0015 + 1.00883 w_k10_HSBC_s001
                                 + 0.99817 w_k10_VOD_s001
                                 + 0.936652 w_k10_BP_s001
                                 + 1.03813 w_k10_GSK_s001
                                 + 1.02586 w_k10_AZN_s001
                                 + 1.0342 w_k10_RIO_s001
                                 + 0.990666 w_k10_BG_s001
                                 + 1.01696 w_k10_TSCO_s001
                                 + 0.937541 w_k10_AAL_s001
                                 + 1.03925 w_k10_PRU_s001 >= 1
 z_k10_s0016_constraint:         var_10 + z_k10_s0016
                                 + 0.794504 w_k10_HSBC_s001
                                 + 0.951707 w_k10_VOD_s001
                                 + 0.798779 w_k10_BP_s001
                                 + 0.809598 w_k10_GSK_s001
                                 + 0.918008 w_k10_AZN_s001
                                 + 0.800599 w_k10_RIO_s001
                                 + 0.854895 w_k10_BG_s001
                                 + 0.717318 w_k10_TSCO_s001
                                 + 0.934574 w_k10_AAL_s001
                                 + 1.066 w_k10_PRU_s001 >= 1
 z_k10_s0017_constraint:         var_10 + z_k10_s0017 + 1.01615 w_k10_HSBC_s001
                                 + 1.2599 w_k10_VOD_s001
                                 + 0.983538 w_k10_BP_s001
                                 + 0.904704 w_k10_GSK_s001
                                 + 1.1307 w_k10_AZN_s001
                                 + 1.02128 w_k10_RIO_s001
                                 + 0.89356 w_k10_BG_s001
                                 + 1.03191 w_k10_TSCO_s001
                                 + 1.13034 w_k10_AAL_s001
                                 + 1.2406 w_k10_PRU_s001 >= 1
 z_k10_s0020_constraint:         var_10 + z_k10_s0020
                                 + 0.945177 w_k10_HSBC_s002
                                 + 1.10918 w_k10_VOD_s002
                                 + 1.15821 w_k10_BP_s002
                                 + 1.04587 w_k10_GSK_s002
                                 + 1.08416 w_k10_AZN_s002
                                 + 1.08329 w_k10_RIO_s002
                                 + 1.09989 w_k10_BG_s002
                                 + 1.15955 w_k10_TSCO_s002
                                 + 0.940496 w_k10_AAL_s002
                                 + 0.976393 w_k10_PRU_s002 >= 1
 z_k10_s0021_constraint:         var_10 + z_k10_s0021
                                 + 0.767374 w_k10_HSBC_s002
                                 + 1.06927 w_k10_VOD_s002
                                 + 0.895121 w_k10_BP_s002
                                 + 0.947407 w_k10_GSK_s002
                                 + 1.06907 w_k10_AZN_s002
                                 + 0.946723 w_k10_RIO_s002
                                 + 0.922411 w_k10_BG_s002
                                 + 0.882075 w_k10_TSCO_s002
                                 + 1.08168 w_k10_AAL_s002
                                 + 1.06528 w_k10_PRU_s002 >= 1
 z_k10_s0022_constraint:         var_10 + z_k10_s0022 + 1.03885 w_k10_HSBC_s002
                                 + 1.20557 w_k10_VOD_s002
                                 + 1.0437 w_k10_BP_s002
                                 + 1.03276 w_k10_GSK_s002
                                 + 0.98995 w_k10_AZN_s002
                                 + 1.04313 w_k10_RIO_s002
                                 + 0.979925 w_k10_BG_s002
                                 + 1.03169 w_k10_TSCO_s002
                                 + 1.072 w_k10_AAL_s002
                                 + 0.779705 w_k10_PRU_s002 >= 1
 z_k10_s0023_constraint:         var_10 + z_k10_s0023 + 1.24242 w_k10_HSBC_s002
                                 + 1.00127 w_k10_VOD_s002
                                 + 1.16145 w_k10_BP_s002
                                 + 0.945975 w_k10_GSK_s002
                                 + 0.804124 w_k10_AZN_s002
                                 + 1.13745 w_k10_RIO_s002
                                 + 1.14146 w_k10_BG_s002
                                 + 1.0597 w_k10_TSCO_s002
                                 + 1.05766 w_k10_AAL_s002
                                 + 0.806373 w_k10_PRU_s002 >= 1
 z_k10_s0024_constraint:         var_10 + z_k10_s0024 + 1.33716 w_k10_HSBC_s002
                                 + 1.16939 w_k10_VOD_s002
                                 + 1.40827 w_k10_BP_s002
                                 + 1.26122 w_k10_GSK_s002
                                 + 1.09598 w_k10_AZN_s002
                                 + 1.34886 w_k10_RIO_s002
                                 + 1.01877 w_k10_BG_s002
                                 + 1.46039 w_k10_TSCO_s002
                                 + 1.02225 w_k10_AAL_s002
                                 + 0.966037 w_k10_PRU_s002 >= 1
 z_k10_s0025_constraint:         var_10 + z_k10_s0025
                                 + 0.957561 w_k10_HSBC_s002
                                 + 1.00731 w_k10_VOD_s002
                                 + 0.754887 w_k10_BP_s002
                                 + 0.867413 w_k10_GSK_s002
                                 + 0.87398 w_k10_AZN_s002
                                 + 0.96923 w_k10_RIO_s002
                                 + 1.07661 w_k10_BG_s002
                                 + 0.790943 w_k10_TSCO_s002
                                 + 0.824877 w_k10_AAL_s002
                                 + 0.621793 w_k10_PRU_s002 >= 1
 z_k10_s0026_constraint:         var_10 + z_k10_s0026
                                 + 0.940334 w_k10_HSBC_s002
                                 + 1.04007 w_k10_VOD_s002
                                 + 0.859587 w_k10_BP_s002
                                 + 0.922806 w_k10_GSK_s002
                                 + 0.875496 w_k10_AZN_s002
                                 + 0.829808 w_k10_RIO_s002
                                 + 1.01787 w_k10_BG_s002
                                 + 0.756158 w_k10_TSCO_s002
                                 + 1.05467 w_k10_AAL_s002
                                 + 0.742567 w_k10_PRU_s002 >= 1
 z_k10_s0027_constraint:         var_10 + z_k10_s0027 + 1.18058 w_k10_HSBC_s002
                                 + 1.04771 w_k10_VOD_s002
                                 + 0.861895 w_k10_BP_s002
                                 + 0.888344 w_k10_GSK_s002
                                 + 0.891747 w_k10_AZN_s002
                                 + 0.993287 w_k10_RIO_s002
                                 + 0.997161 w_k10_BG_s002
                                 + 0.822794 w_k10_TSCO_s002
                                 + 0.944124 w_k10_AAL_s002
                                 + 0.684446 w_k10_PRU_s002 >= 1
 z_k10_s0030_constraint:         var_10 + z_k10_s0030
                                 + 0.868783 w_k10_HSBC_s003
                                 + 0.907667 w_k10_VOD_s003
                                 + 0.742843 w_k10_BP_s003
                                 + 0.874827 w_k10_GSK_s003
                                 + 1.07362 w_k10_AZN_s003
                                 + 0.757037 w_k10_RIO_s003
                                 + 0.983589 w_k10_BG_s003
                                 + 0.786827 w_k10_TSCO_s003
                                 + 1.02725 w_k10_AAL_s003
                                 + 0.887443 w_k10_PRU_s003 >= 1
 z_k10_s0031_constraint:         var_10 + z_k10_s0031 + 1.1504 w_k10_HSBC_s003
                                 + 1.00579 w_k10_VOD_s003
                                 + 0.935132 w_k10_BP_s003
                                 + 1.08357 w_k10_GSK_s003
                                 + 1.15564 w_k10_AZN_s003
                                 + 0.894482 w_k10_RIO_s003
                                 + 0.898136 w_k10_BG_s003
                                 + 1.14471 w_k10_TSCO_s003
                                 + 0.882161 w_k10_AAL_s003
                                 + 0.874514 w_k10_PRU_s003 >= 1
 z_k10_s0032_constraint:         var_10 + z_k10_s0032
                                 + 0.813293 w_k10_HSBC_s003
                                 + 1.02031 w_k10_VOD_s003
                                 + 0.770034 w_k10_BP_s003
                                 + 0.906952 w_k10_GSK_s003
                                 + 1.10091 w_k10_AZN_s003
                                 + 0.752127 w_k10_RIO_s003
                                 + 0.87619 w_k10_BG_s003
                                 + 0.778143 w_k10_TSCO_s003
                                 + 1.0006 w_k10_AAL_s003
                                 + 0.894762 w_k10_PRU_s003 >= 1
 z_k10_s0033_constraint:         var_10 + z_k10_s0033
                                 + 0.996221 w_k10_HSBC_s003
                                 + 1.03845 w_k10_VOD_s003
                                 + 1.12317 w_k10_BP_s003
                                 + 1.14557 w_k10_GSK_s003
                                 + 1.22765 w_k10_AZN_s003
                                 + 0.973279 w_k10_RIO_s003
                                 + 0.973142 w_k10_BG_s003
                                 + 1.10043 w_k10_TSCO_s003
                                 + 1.23319 w_k10_AAL_s003
                                 + 1.45516 w_k10_PRU_s003 >= 1
 z_k10_s0034_constraint:         var_10 + z_k10_s0034 + 1.07498 w_k10_HSBC_s003
                                 + 0.76376 w_k10_VOD_s003
                                 + 0.971032 w_k10_BP_s003
                                 + 1.0779 w_k10_GSK_s003
                                 + 0.97475 w_k10_AZN_s003
                                 + 0.981691 w_k10_RIO_s003
                                 + 1.07569 w_k10_BG_s003
                                 + 1.09133 w_k10_TSCO_s003
                                 + 0.940679 w_k10_AAL_s003
                                 + 0.96701 w_k10_PRU_s003 >= 1
 z_k10_s0035_constraint:         var_10 + z_k10_s0035
                                 + 0.966583 w_k10_HSBC_s003
                                 + 0.860164 w_k10_VOD_s003
                                 + 0.895705 w_k10_BP_s003
                                 + 0.87096 w_k10_GSK_s003
                                 + 0.965302 w_k10_AZN_s003
                                 + 0.86368 w_k10_RIO_s003
                                 + 1.00835 w_k10_BG_s003
                                 + 1.01163 w_k10_TSCO_s003
                                 + 1.02341 w_k10_AAL_s003
                                 + 0.971749 w_k10_PRU_s003 >= 1
 z_k10_s0036_constraint:         var_10 + z_k10_s0036 + 1.18557 w_k10_HSBC_s003
                                 + 1.05756 w_k10_VOD_s003
                                 + 1.30368 w_k10_BP_s003
                                 + 1.13886 w_k10_GSK_s003
                                 + 1.24677 w_k10_AZN_s003
                                 + 1.14815 w_k10_RIO_s003
                                 + 1.05334 w_k10_BG_s003
                                 + 1.39503 w_k10_TSCO_s003
                                 + 0.904386 w_k10_AAL_s003
                                 + 1.25208 w_k10_PRU_s003 >= 1
 z_k10_s0037_constraint:         var_10 + z_k10_s0037
                                 + 0.990884 w_k10_HSBC_s003
                                 + 1.01614 w_k10_VOD_s003
                                 + 0.980413 w_k10_BP_s003
                                 + 0.913277 w_k10_GSK_s003
                                 + 1.10096 w_k10_AZN_s003
                                 + 0.869898 w_k10_RIO_s003
                                 + 0.858016 w_k10_BG_s003
                                 + 0.939504 w_k10_TSCO_s003
                                 + 1.1257 w_k10_AAL_s003
                                 + 1.17278 w_k10_PRU_s003 >= 1
 z_k10_s0040_constraint:         var_10 + z_k10_s0040
                                 + 0.998453 w_k10_HSBC_s004
                                 + 1.03736 w_k10_VOD_s004
                                 + 1.08817 w_k10_BP_s004
                                 + 0.894863 w_k10_GSK_s004
                                 + 0.942026 w_k10_AZN_s004
                                 + 1.05508 w_k10_RIO_s004
                                 + 0.949461 w_k10_BG_s004
                                 + 0.855929 w_k10_TSCO_s004
                                 + 1.04307 w_k10_AAL_s004
                                 + 1.12382 w_k10_PRU_s004 >= 1
 z_k10_s0041_constraint:         var_10 + z_k10_s0041 + 1.01667 w_k10_HSBC_s004
                                 + 0.923069 w_k10_VOD_s004
                                 + 0.812266 w_k10_BP_s004
                                 + 0.675569 w_k10_GSK_s004
                                 + 0.845119 w_k10_AZN_s004
                                 + 0.845052 w_k10_RIO_s004
                                 + 1.02278 w_k10_BG_s004
                                 + 0.804955 w_k10_TSCO_s004
                                 + 0.919785 w_k10_AAL_s004
                                 + 0.996058 w_k10_PRU_s004 >= 1
 z_k10_s0042_constraint:         var_10 + z_k10_s0042
                                 + 0.912383 w_k10_HSBC_s004
                                 + 0.991929 w_k10_VOD_s004
                                 + 0.907737 w_k10_BP_s004
                                 + 0.799026 w_k10_GSK_s004
                                 + 0.984128 w_k10_AZN_s004
                                 + 0.906246 w_k10_RIO_s004
                                 + 0.979241 w_k10_BG_s004
                                 + 0.854911 w_k10_TSCO_s004
                                 + 1.01807 w_k10_AAL_s004
                                 + 1.06828 w_k10_PRU_s004 >= 1
 z_k10_s0043_constraint:         var_10 + z_k10_s0043
                                 + 0.830318 w_k10_HSBC_s004
                                 + 1.09283 w_k10_VOD_s004
                                 + 0.751572 w_k10_BP_s004
                                 + 0.783785 w_k10_GSK_s004
                                 + 0.906884 w_k10_AZN_s004
                                 + 0.858059 w_k10_RIO_s004
                                 + 0.8783 w_k10_BG_s004
                                 + 0.729639 w_k10_TSCO_s004
                                 + 0.989784 w_k10_AAL_s004
                                 + 0.938573 w_k10_PRU_s004 >= 1
 z_k10_s0044_constraint:         var_10 + z_k10_s0044 + 1.10684 w_k10_HSBC_s004
                                 + 1.21948 w_k10_VOD_s004
                                 + 1.36163 w_k10_BP_s004
                                 + 0.927829 w_k10_GSK_s004
                                 + 1.00127 w_k10_AZN_s004
                                 + 1.25351 w_k10_RIO_s004
                                 + 1.05556 w_k10_BG_s004
                                 + 1.24753 w_k10_TSCO_s004
                                 + 1.0342 w_k10_AAL_s004
                                 + 1.29848 w_k10_PRU_s004 >= 1
 z_k10_s0045_constraint:         var_10 + z_k10_s0045 + 1.22155 w_k10_HSBC_s004
                                 + 1.09006 w_k10_VOD_s004
                                 + 1.20143 w_k10_BP_s004
                                 + 1.07722 w_k10_GSK_s004
                                 + 1.03895 w_k10_AZN_s004
                                 + 1.17709 w_k10_RIO_s004
                                 + 1.07785 w_k10_BG_s004
                                 + 1.39652 w_k10_TSCO_s004
                                 + 0.883153 w_k10_AAL_s004
                                 + 1.01269 w_k10_PRU_s004 >= 1
 z_k10_s0046_constraint:         var_10 + z_k10_s0046 + 1.32219 w_k10_HSBC_s004
                                 + 1.08567 w_k10_VOD_s004
                                 + 1.14479 w_k10_BP_s004
                                 + 0.878536 w_k10_GSK_s004
                                 + 1.01917 w_k10_AZN_s004
                                 + 1.05971 w_k10_RIO_s004
                                 + 0.979916 w_k10_BG_s004
                                 + 1.09706 w_k10_TSCO_s004
                                 + 1.06356 w_k10_AAL_s004
                                 + 1.21197 w_k10_PRU_s004 >= 1
 z_k10_s0047_constraint:         var_10 + z_k10_s0047
                                 + 0.953027 w_k10_HSBC_s004
                                 + 1.08891 w_k10_VOD_s004
                                 + 1.05728 w_k10_BP_s004
                                 + 1.02906 w_k10_GSK_s004
                                 + 1.0877 w_k10_AZN_s004
                                 + 1.0353 w_k10_RIO_s004
                                 + 0.918805 w_k10_BG_s004
                                 + 1.09704 w_k10_TSCO_s004
                                 + 0.875253 w_k10_AAL_s004
                                 + 1.01321 w_k10_PRU_s004 >= 1
 z_k10_s0050_constraint:         var_10 + z_k10_s0050 + 1.04404 w_k10_HSBC_s005
                                 + 0.955338 w_k10_VOD_s005
                                 + 1.15342 w_k10_BP_s005
                                 + 1.09349 w_k10_GSK_s005
                                 + 1.04766 w_k10_AZN_s005
                                 + 0.991659 w_k10_RIO_s005
                                 + 1.05306 w_k10_BG_s005
                                 + 1.34443 w_k10_TSCO_s005
                                 + 1.11791 w_k10_AAL_s005
                                 + 1.06314 w_k10_PRU_s005 >= 1
 z_k10_s0051_constraint:         var_10 + z_k10_s0051
                                 + 0.783126 w_k10_HSBC_s005
                                 + 0.76159 w_k10_VOD_s005
                                 + 0.722306 w_k10_BP_s005
                                 + 0.86289 w_k10_GSK_s005
                                 + 0.892631 w_k10_AZN_s005
                                 + 0.746571 w_k10_RIO_s005
                                 + 0.881956 w_k10_BG_s005
                                 + 0.884088 w_k10_TSCO_s005
                                 + 0.949433 w_k10_AAL_s005
                                 + 0.85946 w_k10_PRU_s005 >= 1
 z_k10_s0052_constraint:         var_10 + z_k10_s0052 + 1.03125 w_k10_HSBC_s005
                                 + 0.940342 w_k10_VOD_s005
                                 + 0.838636 w_k10_BP_s005
                                 + 0.765445 w_k10_GSK_s005
                                 + 0.839949 w_k10_AZN_s005
                                 + 0.847876 w_k10_RIO_s005
                                 + 0.893649 w_k10_BG_s005
                                 + 0.910068 w_k10_TSCO_s005
                                 + 1.21223 w_k10_AAL_s005
                                 + 1.08478 w_k10_PRU_s005 >= 1
 z_k10_s0053_constraint:         var_10 + z_k10_s0053
                                 + 0.901476 w_k10_HSBC_s005
                                 + 0.782134 w_k10_VOD_s005
                                 + 0.913967 w_k10_BP_s005
                                 + 0.912531 w_k10_GSK_s005
                                 + 0.803497 w_k10_AZN_s005
                                 + 0.895984 w_k10_RIO_s005
                                 + 1.02724 w_k10_BG_s005
                                 + 0.99808 w_k10_TSCO_s005
                                 + 1.00682 w_k10_AAL_s005
                                 + 0.831659 w_k10_PRU_s005 >= 1
 z_k10_s0054_constraint:         var_10 + z_k10_s0054 + 1.12076 w_k10_HSBC_s005
                                 + 1.18236 w_k10_VOD_s005
                                 + 1.17861 w_k10_BP_s005
                                 + 1.10295 w_k10_GSK_s005
                                 + 1.02433 w_k10_AZN_s005
                                 + 1.15296 w_k10_RIO_s005
                                 + 0.940359 w_k10_BG_s005
                                 + 1.23064 w_k10_TSCO_s005
                                 + 0.940406 w_k10_AAL_s005
                                 + 1.13725 w_k10_PRU_s005 >= 1
 z_k10_s0055_constraint:         var_10 + z_k10_s0055
                                 + 0.963606 w_k10_HSBC_s005
                                 + 0.846373 w_k10_VOD_s005
                                 + 0.98696 w_k10_BP_s005
                                 + 1.06884 w_k10_GSK_s005
                                 + 0.909065 w_k10_AZN_s005
                                 + 0.93273 w_k10_RIO_s005
                                 + 1.08696 w_k10_BG_s005
                                 + 1.19141 w_k10_TSCO_s005
                                 + 0.869534 w_k10_AAL_s005
                                 + 0.830835 w_k10_PRU_s005 >= 1
 z_k10_s0056_constraint:         var_10 + z_k10_s0056 + 1.30396 w_k10_HSBC_s005
                                 + 1.0979 w_k10_VOD_s005
                                 + 1.27991 w_k10_BP_s005
                                 + 1.14315 w_k10_GSK_s005
                                 + 0.957081 w_k10_AZN_s005
                                 + 1.1951 w_k10_RIO_s005
                                 + 1.14395 w_k10_BG_s005
                                 + 1.73817 w_k10_TSCO_s005
                                 + 0.852581 w_k10_AAL_s005
                                 + 1.03082 w_k10_PRU_s005 >= 1
 z_k10_s0057_constraint:         var_10 + z_k10_s0057
                                 + 0.874578 w_k10_HSBC_s005
                                 + 0.831859 w_k10_VOD_s005
                                 + 0.885785 w_k10_BP_s005
                                 + 0.900508 w_k10_GSK_s005
                                 + 0.858406 w_k10_AZN_s005
                                 + 0.914891 w_k10_RIO_s005
                                 + 1.14005 w_k10_BG_s005
                                 + 1.10975 w_k10_TSCO_s005
                                 + 0.82621 w_k10_AAL_s005
                                 + 0.830094 w_k10_PRU_s005 >= 1
 z_k10_s0060_constraint:         var_10 + z_k10_s0060
                                 + 0.903855 w_k10_HSBC_s006
                                 + 0.784067 w_k10_VOD_s006
                                 + 0.878145 w_k10_BP_s006
                                 + 0.874411 w_k10_GSK_s006
                                 + 0.876166 w_k10_AZN_s006
                                 + 0.938003 w_k10_RIO_s006
                                 + 1.04158 w_k10_BG_s006
                                 + 0.95064 w_k10_TSCO_s006
                                 + 0.873454 w_k10_AAL_s006
                                 + 0.903506 w_k10_PRU_s006 >= 1
 z_k10_s0061_constraint:         var_10 + z_k10_s0061 + 1.125 w_k10_HSBC_s006
                                 + 0.990357 w_k10_VOD_s006
                                 + 1.15647 w_k10_BP_s006
                                 + 0.874647 w_k10_GSK_s006
                                 + 0.857516 w_k10_AZN_s006
                                 + 1.16649 w_k10_RIO_s006
                                 + 1.01056 w_k10_BG_s006
                                 + 1.1065 w_k10_TSCO_s006
                                 + 0.849858 w_k10_AAL_s006
                                 + 0.982868 w_k10_PRU_s006 >= 1
 z_k10_s0062_constraint:         var_10 + z_k10_s0062
                                 + 0.883064 w_k10_HSBC_s006
                                 + 0.787082 w_k10_VOD_s006
                                 + 0.929072 w_k10_BP_s006
                                 + 0.839438 w_k10_GSK_s006
                                 + 0.913341 w_k10_AZN_s006
                                 + 0.898767 w_k10_RIO_s006
                                 + 1.14394 w_k10_BG_s006
                                 + 1.05847 w_k10_TSCO_s006
                                 + 0.979296 w_k10_AAL_s006
                                 + 0.906419 w_k10_PRU_s006 >= 1
 z_k10_s0063_constraint:         var_10 + z_k10_s0063
                                 + 0.883429 w_k10_HSBC_s006
                                 + 1.05661 w_k10_VOD_s006
                                 + 1.23712 w_k10_BP_s006
                                 + 1.04126 w_k10_GSK_s006
                                 + 1.05943 w_k10_AZN_s006
                                 + 1.07973 w_k10_RIO_s006
                                 + 0.936642 w_k10_BG_s006
                                 + 1.20987 w_k10_TSCO_s006
                                 + 1.05021 w_k10_AAL_s006
                                 + 1.4814 w_k10_PRU_s006 >= 1
 z_k10_s0064_constraint:         var_10 + z_k10_s0064 + 1.27023 w_k10_HSBC_s006
                                 + 1.07079 w_k10_VOD_s006
                                 + 1.22612 w_k10_BP_s006
                                 + 1.19253 w_k10_GSK_s006
                                 + 1.10688 w_k10_AZN_s006
                                 + 1.15956 w_k10_RIO_s006
                                 + 1.07854 w_k10_BG_s006
                                 + 1.30334 w_k10_TSCO_s006
                                 + 0.959564 w_k10_AAL_s006
                                 + 1.08022 w_k10_PRU_s006 >= 1
 z_k10_s0065_constraint:         var_10 + z_k10_s0065
                                 + 0.803968 w_k10_HSBC_s006
                                 + 1.03168 w_k10_VOD_s006
                                 + 0.783417 w_k10_BP_s006
                                 + 0.832128 w_k10_GSK_s006
                                 + 0.979877 w_k10_AZN_s006
                                 + 0.846625 w_k10_RIO_s006
                                 + 0.884155 w_k10_BG_s006
                                 + 0.789769 w_k10_TSCO_s006
                                 + 0.981562 w_k10_AAL_s006
                                 + 1.02909 w_k10_PRU_s006 >= 1
 z_k10_s0066_constraint:         var_10 + z_k10_s0066
                                 + 0.968231 w_k10_HSBC_s006
                                 + 0.992667 w_k10_VOD_s006
                                 + 0.925611 w_k10_BP_s006
                                 + 0.76553 w_k10_GSK_s006
                                 + 0.96256 w_k10_AZN_s006
                                 + 0.883699 w_k10_RIO_s006
                                 + 0.886558 w_k10_BG_s006
                                 + 0.923946 w_k10_TSCO_s006
                                 + 1.17532 w_k10_AAL_s006
                                 + 1.14633 w_k10_PRU_s006 >= 1
 z_k10_s0067_constraint:         var_10 + z_k10_s0067 + 1.06387 w_k10_HSBC_s006
                                 + 0.984857 w_k10_VOD_s006
                                 + 1.41673 w_k10_BP_s006
                                 + 1.11205 w_k10_GSK_s006
                                 + 0.954102 w_k10_AZN_s006
                                 + 1.23669 w_k10_RIO_s006
                                 + 1.16579 w_k10_BG_s006
                                 + 1.26992 w_k10_TSCO_s006
                                 + 1.14992 w_k10_AAL_s006
                                 + 1.10468 w_k10_PRU_s006 >= 1
 z_k10_s0070_constraint:         var_10 + z_k10_s0070 + 1.01174 w_k10_HSBC_s007
                                 + 0.899977 w_k10_VOD_s007
                                 + 0.921495 w_k10_BP_s007
                                 + 0.997814 w_k10_GSK_s007
                                 + 1.01389 w_k10_AZN_s007
                                 + 0.935621 w_k10_RIO_s007
                                 + 1.05315 w_k10_BG_s007
                                 + 1.03317 w_k10_TSCO_s007
                                 + 0.97936 w_k10_AAL_s007
                                 + 0.919659 w_k10_PRU_s007 >= 1
 z_k10_s0071_constraint:         var_10 + z_k10_s0071 + 1.13207 w_k10_HSBC_s007
                                 + 0.972137 w_k10_VOD_s007
                                 + 0.896313 w_k10_BP_s007
                                 + 0.942837 w_k10_GSK_s007
                                 + 1.0903 w_k10_AZN_s007
                                 + 0.842795 w_k10_RIO_s007
                                 + 0.801757 w_k10_BG_s007
                                 + 0.857038 w_k10_TSCO_s007
                                 + 1.08331 w_k10_AAL_s007
                                 + 0.92204 w_k10_PRU_s007 >= 1
 z_k10_s0072_constraint:         var_10 + z_k10_s0072
                                 + 0.886773 w_k10_HSBC_s007
                                 + 0.951709 w_k10_VOD_s007
                                 + 0.828096 w_k10_BP_s007
                                 + 0.945138 w_k10_GSK_s007
                                 + 0.937052 w_k10_AZN_s007
                                 + 0.784879 w_k10_RIO_s007
                                 + 0.891682 w_k10_BG_s007
                                 + 0.76267 w_k10_TSCO_s007
                                 + 1.11444 w_k10_AAL_s007
                                 + 0.946211 w_k10_PRU_s007 >= 1
 z_k10_s0073_constraint:         var_10 + z_k10_s0073
                                 + 0.875855 w_k10_HSBC_s007
                                 + 0.916023 w_k10_VOD_s007
                                 + 0.852083 w_k10_BP_s007
                                 + 0.959532 w_k10_GSK_s007
                                 + 1.05026 w_k10_AZN_s007
                                 + 0.855976 w_k10_RIO_s007
                                 + 0.945345 w_k10_BG_s007
                                 + 0.788143 w_k10_TSCO_s007
                                 + 0.954099 w_k10_AAL_s007
                                 + 0.791697 w_k10_PRU_s007 >= 1
 z_k10_s0074_constraint:         var_10 + z_k10_s0074 + 1.24405 w_k10_HSBC_s007
                                 + 0.915163 w_k10_VOD_s007
                                 + 1.03974 w_k10_BP_s007
                                 + 1.1202 w_k10_GSK_s007
                                 + 0.984845 w_k10_AZN_s007
                                 + 1.03386 w_k10_RIO_s007
                                 + 0.994242 w_k10_BG_s007
                                 + 1.03812 w_k10_TSCO_s007
                                 + 1.04604 w_k10_AAL_s007
                                 + 1.00873 w_k10_PRU_s007 >= 1
 z_k10_s0075_constraint:         var_10 + z_k10_s0075 + 1.22156 w_k10_HSBC_s007
                                 + 1.05676 w_k10_VOD_s007
                                 + 1.29694 w_k10_BP_s007
                                 + 1.22955 w_k10_GSK_s007
                                 + 1.02302 w_k10_AZN_s007
                                 + 1.28327 w_k10_RIO_s007
                                 + 1.02287 w_k10_BG_s007
                                 + 1.21074 w_k10_TSCO_s007
                                 + 0.984098 w_k10_AAL_s007
                                 + 1.06009 w_k10_PRU_s007 >= 1
 z_k10_s0076_constraint:         var_10 + z_k10_s0076
                                 + 0.859917 w_k10_HSBC_s007
                                 + 1.05403 w_k10_VOD_s007
                                 + 0.968944 w_k10_BP_s007
                                 + 1.08588 w_k10_GSK_s007
                                 + 1.04047 w_k10_AZN_s007
                                 + 0.950955 w_k10_RIO_s007
                                 + 0.956161 w_k10_BG_s007
                                 + 0.970574 w_k10_TSCO_s007
                                 + 1.15128 w_k10_AAL_s007
                                 + 1.18022 w_k10_PRU_s007 >= 1
 z_k10_s0077_constraint:         var_10 + z_k10_s0077
                                 + 0.835309 w_k10_HSBC_s007
                                 + 1.00262 w_k10_VOD_s007
                                 + 0.719421 w_k10_BP_s007
                                 + 0.802158 w_k10_GSK_s007
                                 + 0.897729 w_k10_AZN_s007
                                 + 0.726888 w_k10_RIO_s007
                                 + 0.926389 w_k10_BG_s007
                                 + 0.770106 w_k10_TSCO_s007
                                 + 1.03173 w_k10_AAL_s007
                                 + 0.933065 w_k10_PRU_s007 >= 1
 z_k10_s0100_constraint:         var_10 + z_k10_s0100 + 1.30026 w_k10_HSBC_s010
                                 + 1.37941 w_k10_VOD_s010
                                 + 1.51406 w_k10_BP_s010
                                 + 1.31425 w_k10_GSK_s010
                                 + 1.11316 w_k10_AZN_s010
                                 + 1.33474 w_k10_RIO_s010
                                 + 1.02165 w_k10_BG_s010
                                 + 1.43009 w_k10_TSCO_s010
                                 + 1.02407 w_k10_AAL_s010
                                 + 1.35795 w_k10_PRU_s010 >= 1
 z_k10_s0101_constraint:         var_10 + z_k10_s0101
                                 + 0.878592 w_k10_HSBC_s010
                                 + 1.14278 w_k10_VOD_s010
                                 + 1.14627 w_k10_BP_s010
                                 + 1.21171 w_k10_GSK_s010
                                 + 1.21189 w_k10_AZN_s010
                                 + 1.06821 w_k10_RIO_s010
                                 + 0.977614 w_k10_BG_s010
                                 + 1.36678 w_k10_TSCO_s010
                                 + 1.06327 w_k10_AAL_s010
                                 + 1.21374 w_k10_PRU_s010 >= 1
 z_k10_s0102_constraint:         var_10 + z_k10_s0102 + 1.31824 w_k10_HSBC_s010
                                 + 1.1824 w_k10_VOD_s010
                                 + 1.08051 w_k10_BP_s010
                                 + 1.03144 w_k10_GSK_s010
                                 + 0.949839 w_k10_AZN_s010
                                 + 1.14591 w_k10_RIO_s010
                                 + 1.08224 w_k10_BG_s010
                                 + 1.15236 w_k10_TSCO_s010
                                 + 0.844429 w_k10_AAL_s010
                                 + 0.861265 w_k10_PRU_s010 >= 1
 z_k10_s0103_constraint:         var_10 + z_k10_s0103 + 1.18152 w_k10_HSBC_s010
                                 + 0.935148 w_k10_VOD_s010
                                 + 0.907649 w_k10_BP_s010
                                 + 1.13233 w_k10_GSK_s010
                                 + 0.923581 w_k10_AZN_s010
                                 + 1.02857 w_k10_RIO_s010
                                 + 1.13598 w_k10_BG_s010
                                 + 1.03706 w_k10_TSCO_s010
                                 + 0.984075 w_k10_AAL_s010
                                 + 0.884331 w_k10_PRU_s010 >= 1
 z_k10_s0104_constraint:         var_10 + z_k10_s0104 + 1.34655 w_k10_HSBC_s010
                                 + 1.11281 w_k10_VOD_s010
                                 + 1.28826 w_k10_BP_s010
                                 + 1.21975 w_k10_GSK_s010
                                 + 1.05238 w_k10_AZN_s010
                                 + 1.20165 w_k10_RIO_s010
                                 + 1.09063 w_k10_BG_s010
                                 + 1.3282 w_k10_TSCO_s010
                                 + 1.10176 w_k10_AAL_s010
                                 + 1.11261 w_k10_PRU_s010 >= 1
 z_k10_s0105_constraint:         var_10 + z_k10_s0105
                                 + 0.874593 w_k10_HSBC_s010
                                 + 0.973404 w_k10_VOD_s010
                                 + 0.859969 w_k10_BP_s010
                                 + 0.921096 w_k10_GSK_s010
                                 + 1.02319 w_k10_AZN_s010
                                 + 0.825787 w_k10_RIO_s010
                                 + 0.895144 w_k10_BG_s010
                                 + 0.885413 w_k10_TSCO_s010
                                 + 1.03341 w_k10_AAL_s010
                                 + 0.877827 w_k10_PRU_s010 >= 1
 z_k10_s0106_constraint:         var_10 + z_k10_s0106 + 1.05997 w_k10_HSBC_s010
                                 + 0.887605 w_k10_VOD_s010
                                 + 0.688399 w_k10_BP_s010
                                 + 0.937426 w_k10_GSK_s010
                                 + 1.02681 w_k10_AZN_s010
                                 + 0.823656 w_k10_RIO_s010
                                 + 0.98237 w_k10_BG_s010
                                 + 0.958827 w_k10_TSCO_s010
                                 + 1.01617 w_k10_AAL_s010
                                 + 0.797187 w_k10_PRU_s010 >= 1
 z_k10_s0107_constraint:         var_10 + z_k10_s0107 + 1.03852 w_k10_HSBC_s010
                                 + 1.07032 w_k10_VOD_s010
                                 + 1.05759 w_k10_BP_s010
                                 + 0.974943 w_k10_GSK_s010
                                 + 1.03049 w_k10_AZN_s010
                                 + 1.07777 w_k10_RIO_s010
                                 + 0.889005 w_k10_BG_s010
                                 + 1.10311 w_k10_TSCO_s010
                                 + 1.03848 w_k10_AAL_s010
                                 + 1.22739 w_k10_PRU_s010 >= 1
 z_k10_s0110_constraint:         var_10 + z_k10_s0110 + 1.10187 w_k10_HSBC_s011
                                 + 0.983698 w_k10_VOD_s011
                                 + 1.11187 w_k10_BP_s011
                                 + 1.03252 w_k10_GSK_s011
                                 + 0.981063 w_k10_AZN_s011
                                 + 1.15816 w_k10_RIO_s011
                                 + 1.19994 w_k10_BG_s011
                                 + 1.03791 w_k10_TSCO_s011
                                 + 1.05269 w_k10_AAL_s011
                                 + 0.957578 w_k10_PRU_s011 >= 1
 z_k10_s0111_constraint:         var_10 + z_k10_s0111
                                 + 0.830268 w_k10_HSBC_s011
                                 + 1.0465 w_k10_VOD_s011
                                 + 1.03008 w_k10_BP_s011
                                 + 0.995856 w_k10_GSK_s011
                                 + 1.17287 w_k10_AZN_s011
                                 + 1.00484 w_k10_RIO_s011
                                 + 0.894289 w_k10_BG_s011
                                 + 0.976447 w_k10_TSCO_s011
                                 + 1.39731 w_k10_AAL_s011
                                 + 1.22519 w_k10_PRU_s011 >= 1
 z_k10_s0112_constraint:         var_10 + z_k10_s0112
                                 + 0.816112 w_k10_HSBC_s011
                                 + 1.15594 w_k10_VOD_s011
                                 + 1.09114 w_k10_BP_s011
                                 + 1.10529 w_k10_GSK_s011
                                 + 1.13521 w_k10_AZN_s011
                                 + 1.08076 w_k10_RIO_s011
                                 + 1.01304 w_k10_BG_s011
                                 + 1.04126 w_k10_TSCO_s011
                                 + 1.03097 w_k10_AAL_s011
                                 + 1.11891 w_k10_PRU_s011 >= 1
 z_k10_s0113_constraint:         var_10 + z_k10_s0113 + 1.07971 w_k10_HSBC_s011
                                 + 1.1733 w_k10_VOD_s011
                                 + 1.17323 w_k10_BP_s011
                                 + 1.08177 w_k10_GSK_s011
                                 + 1.18465 w_k10_AZN_s011
                                 + 1.36642 w_k10_RIO_s011
                                 + 0.986663 w_k10_BG_s011
                                 + 1.21732 w_k10_TSCO_s011
                                 + 1.00463 w_k10_AAL_s011
                                 + 1.34866 w_k10_PRU_s011 >= 1
 z_k10_s0114_constraint:         var_10 + z_k10_s0114
                                 + 0.893646 w_k10_HSBC_s011
                                 + 0.934941 w_k10_VOD_s011
                                 + 0.822375 w_k10_BP_s011
                                 + 0.876246 w_k10_GSK_s011
                                 + 0.869632 w_k10_AZN_s011
                                 + 0.974327 w_k10_RIO_s011
                                 + 0.96977 w_k10_BG_s011
                                 + 0.817675 w_k10_TSCO_s011
                                 + 0.918275 w_k10_AAL_s011
                                 + 0.884347 w_k10_PRU_s011 >= 1
 z_k10_s0115_constraint:         var_10 + z_k10_s0115 + 1.10541 w_k10_HSBC_s011
                                 + 1.02043 w_k10_VOD_s011
                                 + 1.41869 w_k10_BP_s011
                                 + 1.28043 w_k10_GSK_s011
                                 + 1.1494 w_k10_AZN_s011
                                 + 1.28131 w_k10_RIO_s011
                                 + 1.03496 w_k10_BG_s011
                                 + 1.44938 w_k10_TSCO_s011
                                 + 1.18399 w_k10_AAL_s011
                                 + 1.21174 w_k10_PRU_s011 >= 1
 z_k10_s0116_constraint:         var_10 + z_k10_s0116
                                 + 0.879248 w_k10_HSBC_s011
                                 + 0.908838 w_k10_VOD_s011
                                 + 0.782733 w_k10_BP_s011
                                 + 0.932041 w_k10_GSK_s011
                                 + 1.01679 w_k10_AZN_s011
                                 + 0.906324 w_k10_RIO_s011
                                 + 0.869339 w_k10_BG_s011
                                 + 0.756979 w_k10_TSCO_s011
                                 + 1.16903 w_k10_AAL_s011
                                 + 1.12233 w_k10_PRU_s011 >= 1
 z_k10_s0117_constraint:         var_10 + z_k10_s0117 + 1.0986 w_k10_HSBC_s011
                                 + 0.999364 w_k10_VOD_s011
                                 + 0.966458 w_k10_BP_s011
                                 + 0.940065 w_k10_GSK_s011
                                 + 0.959722 w_k10_AZN_s011
                                 + 1.03761 w_k10_RIO_s011
                                 + 1.11599 w_k10_BG_s011
                                 + 0.984116 w_k10_TSCO_s011
                                 + 0.93612 w_k10_AAL_s011
                                 + 0.913863 w_k10_PRU_s011 >= 1
 z_k10_s0120_constraint:         var_10 + z_k10_s0120 + 1.03667 w_k10_HSBC_s012
                                 + 1.09754 w_k10_VOD_s012
                                 + 1.18755 w_k10_BP_s012
                                 + 1.06414 w_k10_GSK_s012
                                 + 1.07172 w_k10_AZN_s012
                                 + 1.01479 w_k10_RIO_s012
                                 + 0.946434 w_k10_BG_s012
                                 + 1.20716 w_k10_TSCO_s012
                                 + 1.22219 w_k10_AAL_s012
                                 + 1.14783 w_k10_PRU_s012 >= 1
 z_k10_s0121_constraint:         var_10 + z_k10_s0121 + 1.29488 w_k10_HSBC_s012
                                 + 1.11584 w_k10_VOD_s012
                                 + 1.09555 w_k10_BP_s012
                                 + 1.21335 w_k10_GSK_s012
                                 + 1.11762 w_k10_AZN_s012
                                 + 1.05674 w_k10_RIO_s012
                                 + 0.947377 w_k10_BG_s012
                                 + 1.18742 w_k10_TSCO_s012
                                 + 0.9312 w_k10_AAL_s012
                                 + 1.06556 w_k10_PRU_s012 >= 1
 z_k10_s0122_constraint:         var_10 + z_k10_s0122
                                 + 0.965878 w_k10_HSBC_s012
                                 + 0.946477 w_k10_VOD_s012
                                 + 0.713084 w_k10_BP_s012
                                 + 0.899588 w_k10_GSK_s012
                                 + 0.933635 w_k10_AZN_s012
                                 + 0.854049 w_k10_RIO_s012
                                 + 0.922196 w_k10_BG_s012
                                 + 1.01401 w_k10_TSCO_s012
                                 + 0.786022 w_k10_AAL_s012
                                 + 0.736732 w_k10_PRU_s012 >= 1
 z_k10_s0123_constraint:         var_10 + z_k10_s0123 + 1.14332 w_k10_HSBC_s012
                                 + 0.936675 w_k10_VOD_s012
                                 + 0.93301 w_k10_BP_s012
                                 + 0.986637 w_k10_GSK_s012
                                 + 0.926335 w_k10_AZN_s012
                                 + 0.984301 w_k10_RIO_s012
                                 + 1.08846 w_k10_BG_s012
                                 + 0.985065 w_k10_TSCO_s012
                                 + 0.905956 w_k10_AAL_s012
                                 + 0.791657 w_k10_PRU_s012 >= 1
 z_k10_s0124_constraint:         var_10 + z_k10_s0124
                                 + 0.980629 w_k10_HSBC_s012
                                 + 1.01208 w_k10_VOD_s012
                                 + 0.975262 w_k10_BP_s012
                                 + 1.1572 w_k10_GSK_s012
                                 + 1.1028 w_k10_AZN_s012
                                 + 1.04475 w_k10_RIO_s012
                                 + 1.02993 w_k10_BG_s012
                                 + 1.05222 w_k10_TSCO_s012
                                 + 0.908961 w_k10_AAL_s012
                                 + 0.920232 w_k10_PRU_s012 >= 1
 z_k10_s0125_constraint:         var_10 + z_k10_s0125 + 1.0095 w_k10_HSBC_s012
                                 + 1.04521 w_k10_VOD_s012
                                 + 1.2248 w_k10_BP_s012 + 1.1264 w_k10_GSK_s012
                                 + 1.02008 w_k10_AZN_s012
                                 + 1.09259 w_k10_RIO_s012
                                 + 1.01598 w_k10_BG_s012
                                 + 1.18163 w_k10_TSCO_s012
                                 + 0.920336 w_k10_AAL_s012
                                 + 1.1548 w_k10_PRU_s012 >= 1
 z_k10_s0126_constraint:         var_10 + z_k10_s0126
                                 + 0.878145 w_k10_HSBC_s012
                                 + 0.964623 w_k10_VOD_s012
                                 + 0.98971 w_k10_BP_s012
                                 + 0.974865 w_k10_GSK_s012
                                 + 1.07006 w_k10_AZN_s012
                                 + 0.862288 w_k10_RIO_s012
                                 + 0.848112 w_k10_BG_s012
                                 + 1.07621 w_k10_TSCO_s012
                                 + 0.982428 w_k10_AAL_s012
                                 + 1.21438 w_k10_PRU_s012 >= 1
 z_k10_s0127_constraint:         var_10 + z_k10_s0127
                                 + 0.929792 w_k10_HSBC_s012
                                 + 0.901423 w_k10_VOD_s012
                                 + 0.737798 w_k10_BP_s012
                                 + 0.829489 w_k10_GSK_s012
                                 + 0.973898 w_k10_AZN_s012
                                 + 0.794916 w_k10_RIO_s012
                                 + 0.906641 w_k10_BG_s012
                                 + 0.823658 w_k10_TSCO_s012
                                 + 1.16909 w_k10_AAL_s012
                                 + 1.05562 w_k10_PRU_s012 >= 1
 z_k10_s0130_constraint:         var_10 + z_k10_s0130 + 1.02595 w_k10_HSBC_s013
                                 + 0.901923 w_k10_VOD_s013
                                 + 1.03617 w_k10_BP_s013
                                 + 1.09853 w_k10_GSK_s013
                                 + 1.08311 w_k10_AZN_s013
                                 + 1.00216 w_k10_RIO_s013
                                 + 1.07202 w_k10_BG_s013
                                 + 1.07952 w_k10_TSCO_s013
                                 + 1.05102 w_k10_AAL_s013
                                 + 1.01289 w_k10_PRU_s013 >= 1
 z_k10_s0131_constraint:         var_10 + z_k10_s0131
                                 + 0.891319 w_k10_HSBC_s013
                                 + 0.891918 w_k10_VOD_s013
                                 + 0.924094 w_k10_BP_s013
                                 + 0.9797 w_k10_GSK_s013
                                 + 1.09574 w_k10_AZN_s013
                                 + 0.778824 w_k10_RIO_s013
                                 + 0.930269 w_k10_BG_s013
                                 + 0.78367 w_k10_TSCO_s013
                                 + 1.35429 w_k10_AAL_s013
                                 + 1.07764 w_k10_PRU_s013 >= 1
 z_k10_s0132_constraint:         var_10 + z_k10_s0132
                                 + 0.923871 w_k10_HSBC_s013
                                 + 1.01624 w_k10_VOD_s013
                                 + 0.788286 w_k10_BP_s013
                                 + 0.785609 w_k10_GSK_s013
                                 + 0.893791 w_k10_AZN_s013
                                 + 0.79717 w_k10_RIO_s013
                                 + 0.930078 w_k10_BG_s013
                                 + 0.726342 w_k10_TSCO_s013
                                 + 0.857064 w_k10_AAL_s013
                                 + 0.785353 w_k10_PRU_s013 >= 1
 z_k10_s0133_constraint:         var_10 + z_k10_s0133 + 1.17565 w_k10_HSBC_s013
                                 + 0.939756 w_k10_VOD_s013
                                 + 0.919464 w_k10_BP_s013
                                 + 0.859478 w_k10_GSK_s013
                                 + 0.976134 w_k10_AZN_s013
                                 + 0.885717 w_k10_RIO_s013
                                 + 0.968366 w_k10_BG_s013
                                 + 1.1203 w_k10_TSCO_s013
                                 + 0.971478 w_k10_AAL_s013
                                 + 0.895551 w_k10_PRU_s013 >= 1
 z_k10_s0134_constraint:         var_10 + z_k10_s0134 + 1.13874 w_k10_HSBC_s013
                                 + 0.977873 w_k10_VOD_s013
                                 + 1.03379 w_k10_BP_s013
                                 + 0.993386 w_k10_GSK_s013
                                 + 1.04122 w_k10_AZN_s013
                                 + 0.965795 w_k10_RIO_s013
                                 + 0.992681 w_k10_BG_s013
                                 + 1.16288 w_k10_TSCO_s013
                                 + 0.84437 w_k10_AAL_s013
                                 + 0.808892 w_k10_PRU_s013 >= 1
 z_k10_s0135_constraint:         var_10 + z_k10_s0135 + 1.4145 w_k10_HSBC_s013
                                 + 0.969929 w_k10_VOD_s013
                                 + 1.37159 w_k10_BP_s013
                                 + 1.22759 w_k10_GSK_s013
                                 + 1.1474 w_k10_AZN_s013
                                 + 1.17923 w_k10_RIO_s013
                                 + 1.23297 w_k10_BG_s013
                                 + 1.62613 w_k10_TSCO_s013
                                 + 0.899537 w_k10_AAL_s013
                                 + 0.834588 w_k10_PRU_s013 >= 1
 z_k10_s0136_constraint:         var_10 + z_k10_s0136
                                 + 0.929397 w_k10_HSBC_s013
                                 + 0.847578 w_k10_VOD_s013
                                 + 0.96603 w_k10_BP_s013
                                 + 0.851375 w_k10_GSK_s013
                                 + 0.920673 w_k10_AZN_s013
                                 + 0.794479 w_k10_RIO_s013
                                 + 0.998477 w_k10_BG_s013
                                 + 0.89443 w_k10_TSCO_s013
                                 + 1.04081 w_k10_AAL_s013
                                 + 0.856178 w_k10_PRU_s013 >= 1
 z_k10_s0137_constraint:         var_10 + z_k10_s0137 + 1.20961 w_k10_HSBC_s013
                                 + 0.99671 w_k10_VOD_s013
                                 + 1.30118 w_k10_BP_s013
                                 + 1.05969 w_k10_GSK_s013
                                 + 1.05078 w_k10_AZN_s013
                                 + 1.07201 w_k10_RIO_s013
                                 + 1.11035 w_k10_BG_s013
                                 + 1.32787 w_k10_TSCO_s013
                                 + 1.05481 w_k10_AAL_s013
                                 + 1.02972 w_k10_PRU_s013 >= 1
 z_k10_s0140_constraint:         var_10 + z_k10_s0140 + 1.22239 w_k10_HSBC_s014
                                 + 1.07234 w_k10_VOD_s014
                                 + 1.23522 w_k10_BP_s014
                                 + 1.03487 w_k10_GSK_s014
                                 + 1.0279 w_k10_AZN_s014
                                 + 1.08525 w_k10_RIO_s014
                                 + 1.01525 w_k10_BG_s014
                                 + 1.06575 w_k10_TSCO_s014
                                 + 1.0752 w_k10_AAL_s014
                                 + 1.24492 w_k10_PRU_s014 >= 1
 z_k10_s0141_constraint:         var_10 + z_k10_s0141
                                 + 0.864386 w_k10_HSBC_s014
                                 + 1.00001 w_k10_VOD_s014
                                 + 0.825041 w_k10_BP_s014
                                 + 0.853381 w_k10_GSK_s014
                                 + 0.984254 w_k10_AZN_s014
                                 + 0.779203 w_k10_RIO_s014
                                 + 0.846771 w_k10_BG_s014
                                 + 0.732355 w_k10_TSCO_s014
                                 + 1.11726 w_k10_AAL_s014
                                 + 1.11117 w_k10_PRU_s014 >= 1
 z_k10_s0142_constraint:         var_10 + z_k10_s0142 + 1.02874 w_k10_HSBC_s014
                                 + 0.985796 w_k10_VOD_s014
                                 + 0.840113 w_k10_BP_s014
                                 + 0.967898 w_k10_GSK_s014
                                 + 0.802509 w_k10_AZN_s014
                                 + 0.881676 w_k10_RIO_s014
                                 + 0.949351 w_k10_BG_s014
                                 + 0.796394 w_k10_TSCO_s014
                                 + 0.822332 w_k10_AAL_s014
                                 + 0.92223 w_k10_PRU_s014 >= 1
 z_k10_s0143_constraint:         var_10 + z_k10_s0143
                                 + 0.869836 w_k10_HSBC_s014
                                 + 0.892269 w_k10_VOD_s014
                                 + 0.837039 w_k10_BP_s014
                                 + 0.990569 w_k10_GSK_s014
                                 + 0.934637 w_k10_AZN_s014
                                 + 0.87227 w_k10_RIO_s014
                                 + 1.05689 w_k10_BG_s014
                                 + 0.850407 w_k10_TSCO_s014
                                 + 1.04479 w_k10_AAL_s014
                                 + 0.998581 w_k10_PRU_s014 >= 1
 z_k10_s0144_constraint:         var_10 + z_k10_s0144 + 1.20016 w_k10_HSBC_s014
                                 + 0.897411 w_k10_VOD_s014
                                 + 0.893995 w_k10_BP_s014
                                 + 0.913001 w_k10_GSK_s014
                                 + 0.842821 w_k10_AZN_s014
                                 + 0.982676 w_k10_RIO_s014
                                 + 1.12702 w_k10_BG_s014
                                 + 1.00208 w_k10_TSCO_s014
                                 + 0.950707 w_k10_AAL_s014
                                 + 0.963103 w_k10_PRU_s014 >= 1
 z_k10_s0145_constraint:         var_10 + z_k10_s0145
                                 + 0.913439 w_k10_HSBC_s014
                                 + 1.00265 w_k10_VOD_s014
                                 + 1.03664 w_k10_BP_s014
                                 + 1.0551 w_k10_GSK_s014
                                 + 0.905646 w_k10_AZN_s014
                                 + 1.03372 w_k10_RIO_s014
                                 + 0.977084 w_k10_BG_s014
                                 + 0.907755 w_k10_TSCO_s014
                                 + 1.01624 w_k10_AAL_s014
                                 + 1.08562 w_k10_PRU_s014 >= 1
 z_k10_s0146_constraint:         var_10 + z_k10_s0146 + 1.01599 w_k10_HSBC_s014
                                 + 1.10961 w_k10_VOD_s014
                                 + 1.07343 w_k10_BP_s014
                                 + 1.00094 w_k10_GSK_s014
                                 + 1.05515 w_k10_AZN_s014
                                 + 0.906573 w_k10_RIO_s014
                                 + 0.969455 w_k10_BG_s014
                                 + 1.07977 w_k10_TSCO_s014
                                 + 1.17457 w_k10_AAL_s014
                                 + 1.26294 w_k10_PRU_s014 >= 1
 z_k10_s0147_constraint:         var_10 + z_k10_s0147 + 1.22849 w_k10_HSBC_s014
                                 + 0.913133 w_k10_VOD_s014
                                 + 1.16899 w_k10_BP_s014
                                 + 1.00137 w_k10_GSK_s014
                                 + 0.905689 w_k10_AZN_s014
                                 + 1.11287 w_k10_RIO_s014
                                 + 1.20377 w_k10_BG_s014
                                 + 1.16144 w_k10_TSCO_s014
                                 + 0.962176 w_k10_AAL_s014
                                 + 1.14874 w_k10_PRU_s014 >= 1
 z_k10_s0150_constraint:         var_10 + z_k10_s0150
                                 + 0.974023 w_k10_HSBC_s015
                                 + 0.935398 w_k10_VOD_s015
                                 + 0.784923 w_k10_BP_s015
                                 + 0.918379 w_k10_GSK_s015
                                 + 1.03298 w_k10_AZN_s015
                                 + 0.810726 w_k10_RIO_s015
                                 + 0.906762 w_k10_BG_s015
                                 + 0.922266 w_k10_TSCO_s015
                                 + 0.895643 w_k10_AAL_s015
                                 + 0.806807 w_k10_PRU_s015 >= 1
 z_k10_s0151_constraint:         var_10 + z_k10_s0151 + 1.0154 w_k10_HSBC_s015
                                 + 1.16783 w_k10_VOD_s015
                                 + 1.13231 w_k10_BP_s015
                                 + 1.3152 w_k10_GSK_s015
                                 + 1.19256 w_k10_AZN_s015
                                 + 1.1398 w_k10_RIO_s015
                                 + 0.993516 w_k10_BG_s015
                                 + 1.21361 w_k10_TSCO_s015
                                 + 1.16475 w_k10_AAL_s015
                                 + 1.22961 w_k10_PRU_s015 >= 1
 z_k10_s0152_constraint:         var_10 + z_k10_s0152 + 1.10295 w_k10_HSBC_s015
                                 + 1.06985 w_k10_VOD_s015
                                 + 1.18647 w_k10_BP_s015
                                 + 1.12206 w_k10_GSK_s015
                                 + 0.978506 w_k10_AZN_s015
                                 + 1.1516 w_k10_RIO_s015
                                 + 1.12524 w_k10_BG_s015
                                 + 1.21383 w_k10_TSCO_s015
                                 + 0.91555 w_k10_AAL_s015
                                 + 0.918944 w_k10_PRU_s015 >= 1
 z_k10_s0153_constraint:         var_10 + z_k10_s0153
                                 + 0.959893 w_k10_HSBC_s015
                                 + 1.06445 w_k10_VOD_s015
                                 + 0.856963 w_k10_BP_s015
                                 + 1.17271 w_k10_GSK_s015
                                 + 1.1751 w_k10_AZN_s015
                                 + 0.928513 w_k10_RIO_s015
                                 + 0.983662 w_k10_BG_s015
                                 + 0.971078 w_k10_TSCO_s015
                                 + 1.17219 w_k10_AAL_s015
                                 + 1.03001 w_k10_PRU_s015 >= 1
 z_k10_s0154_constraint:         var_10 + z_k10_s0154
                                 + 0.741342 w_k10_HSBC_s015
                                 + 1.05048 w_k10_VOD_s015
                                 + 1.00179 w_k10_BP_s015
                                 + 1.19783 w_k10_GSK_s015
                                 + 1.24498 w_k10_AZN_s015
                                 + 0.944185 w_k10_RIO_s015
                                 + 0.914853 w_k10_BG_s015
                                 + 1.05965 w_k10_TSCO_s015
                                 + 1.13276 w_k10_AAL_s015
                                 + 1.14559 w_k10_PRU_s015 >= 1
 z_k10_s0155_constraint:         var_10 + z_k10_s0155
                                 + 0.995128 w_k10_HSBC_s015
                                 + 0.864089 w_k10_VOD_s015
                                 + 0.857226 w_k10_BP_s015
                                 + 0.954699 w_k10_GSK_s015
                                 + 0.917254 w_k10_AZN_s015
                                 + 0.970488 w_k10_RIO_s015
                                 + 1.07074 w_k10_BG_s015
                                 + 1.0176 w_k10_TSCO_s015
                                 + 1.01187 w_k10_AAL_s015
                                 + 0.773356 w_k10_PRU_s015 >= 1
 z_k10_s0156_constraint:         var_10 + z_k10_s0156
                                 + 0.789461 w_k10_HSBC_s015
                                 + 1.0254 w_k10_VOD_s015
                                 + 0.865168 w_k10_BP_s015
                                 + 0.849779 w_k10_GSK_s015
                                 + 1.0449 w_k10_AZN_s015
                                 + 0.768562 w_k10_RIO_s015
                                 + 0.835475 w_k10_BG_s015
                                 + 0.871211 w_k10_TSCO_s015
                                 + 1.21299 w_k10_AAL_s015
                                 + 1.17564 w_k10_PRU_s015 >= 1
 z_k10_s0157_constraint:         var_10 + z_k10_s0157
                                 + 0.947878 w_k10_HSBC_s015
                                 + 0.928738 w_k10_VOD_s015
                                 + 1.09096 w_k10_BP_s015
                                 + 1.23902 w_k10_GSK_s015
                                 + 1.04916 w_k10_AZN_s015
                                 + 1.029 w_k10_RIO_s015
                                 + 0.948182 w_k10_BG_s015
                                 + 1.32474 w_k10_TSCO_s015
                                 + 1.0074 w_k10_AAL_s015
                                 + 0.986223 w_k10_PRU_s015 >= 1
 z_k10_s0160_constraint:         var_10 + z_k10_s0160
                                 + 0.930165 w_k10_HSBC_s016
                                 + 1.15103 w_k10_VOD_s016
                                 + 1.04414 w_k10_BP_s016
                                 + 1.03785 w_k10_GSK_s016
                                 + 0.849773 w_k10_AZN_s016
                                 + 1.10734 w_k10_RIO_s016
                                 + 1.01766 w_k10_BG_s016
                                 + 0.9369 w_k10_TSCO_s016
                                 + 1.19147 w_k10_AAL_s016
                                 + 1.11712 w_k10_PRU_s016 >= 1
 z_k10_s0161_constraint:         var_10 + z_k10_s0161
                                 + 0.880602 w_k10_HSBC_s016
                                 + 1.02449 w_k10_VOD_s016
                                 + 0.94765 w_k10_BP_s016
                                 + 1.00682 w_k10_GSK_s016
                                 + 0.83204 w_k10_AZN_s016
                                 + 1.11968 w_k10_RIO_s016
                                 + 1.03372 w_k10_BG_s016
                                 + 1.03438 w_k10_TSCO_s016
                                 + 0.835647 w_k10_AAL_s016
                                 + 0.868883 w_k10_PRU_s016 >= 1
 z_k10_s0162_constraint:         var_10 + z_k10_s0162
                                 + 0.746239 w_k10_HSBC_s016
                                 + 0.741996 w_k10_VOD_s016
                                 + 0.784758 w_k10_BP_s016
                                 + 0.797789 w_k10_GSK_s016
                                 + 0.789804 w_k10_AZN_s016
                                 + 0.762574 w_k10_RIO_s016
                                 + 0.891445 w_k10_BG_s016
                                 + 0.675578 w_k10_TSCO_s016
                                 + 1.08774 w_k10_AAL_s016
                                 + 0.874414 w_k10_PRU_s016 >= 1
 z_k10_s0163_constraint:         var_10 + z_k10_s0163 + 1.14261 w_k10_HSBC_s016
                                 + 0.916375 w_k10_VOD_s016
                                 + 1.15975 w_k10_BP_s016
                                 + 1.17762 w_k10_GSK_s016
                                 + 0.99018 w_k10_AZN_s016
                                 + 1.15881 w_k10_RIO_s016
                                 + 1.11399 w_k10_BG_s016
                                 + 1.15222 w_k10_TSCO_s016
                                 + 0.84734 w_k10_AAL_s016
                                 + 0.718308 w_k10_PRU_s016 >= 1
 z_k10_s0164_constraint:         var_10 + z_k10_s0164 + 1.1134 w_k10_HSBC_s016
                                 + 1.0167 w_k10_VOD_s016
                                 + 1.03813 w_k10_BP_s016
                                 + 0.932552 w_k10_GSK_s016
                                 + 0.973806 w_k10_AZN_s016
                                 + 1.00905 w_k10_RIO_s016
                                 + 0.876501 w_k10_BG_s016
                                 + 1.17887 w_k10_TSCO_s016
                                 + 1.16163 w_k10_AAL_s016
                                 + 1.17277 w_k10_PRU_s016 >= 1
 z_k10_s0165_constraint:         var_10 + z_k10_s0165
                                 + 0.922738 w_k10_HSBC_s016
                                 + 0.910261 w_k10_VOD_s016
                                 + 0.9046 w_k10_BP_s016
                                 + 0.846993 w_k10_GSK_s016
                                 + 0.895861 w_k10_AZN_s016
                                 + 0.867196 w_k10_RIO_s016
                                 + 0.950077 w_k10_BG_s016
                                 + 0.90627 w_k10_TSCO_s016
                                 + 0.940202 w_k10_AAL_s016
                                 + 0.752355 w_k10_PRU_s016 >= 1
 z_k10_s0166_constraint:         var_10 + z_k10_s0166
                                 + 0.894965 w_k10_HSBC_s016
                                 + 1.11517 w_k10_VOD_s016
                                 + 1.21669 w_k10_BP_s016
                                 + 1.05745 w_k10_GSK_s016
                                 + 0.993561 w_k10_AZN_s016
                                 + 1.14403 w_k10_RIO_s016
                                 + 0.94562 w_k10_BG_s016
                                 + 0.955641 w_k10_TSCO_s016
                                 + 1.01649 w_k10_AAL_s016
                                 + 1.33805 w_k10_PRU_s016 >= 1
 z_k10_s0167_constraint:         var_10 + z_k10_s0167 + 1.13075 w_k10_HSBC_s016
                                 + 1.04293 w_k10_VOD_s016
                                 + 1.48545 w_k10_BP_s016
                                 + 1.23504 w_k10_GSK_s016
                                 + 1.02418 w_k10_AZN_s016
                                 + 1.33502 w_k10_RIO_s016
                                 + 1.23281 w_k10_BG_s016
                                 + 1.48926 w_k10_TSCO_s016
                                 + 1.01134 w_k10_AAL_s016
                                 + 0.867412 w_k10_PRU_s016 >= 1
 z_k10_s0170_constraint:         var_10 + z_k10_s0170
                                 + 0.837743 w_k10_HSBC_s017
                                 + 0.816543 w_k10_VOD_s017
                                 + 0.712669 w_k10_BP_s017
                                 + 0.705161 w_k10_GSK_s017
                                 + 0.813898 w_k10_AZN_s017
                                 + 0.790685 w_k10_RIO_s017
                                 + 1.065 w_k10_BG_s017
                                 + 0.680816 w_k10_TSCO_s017
                                 + 0.963302 w_k10_AAL_s017
                                 + 0.764813 w_k10_PRU_s017 >= 1
 z_k10_s0171_constraint:         var_10 + z_k10_s0171 + 1.20369 w_k10_HSBC_s017
                                 + 0.733165 w_k10_VOD_s017
                                 + 1.14231 w_k10_BP_s017
                                 + 0.913504 w_k10_GSK_s017
                                 + 0.859659 w_k10_AZN_s017
                                 + 1.04121 w_k10_RIO_s017
                                 + 1.15892 w_k10_BG_s017
                                 + 1.29064 w_k10_TSCO_s017
                                 + 0.906214 w_k10_AAL_s017
                                 + 0.84951 w_k10_PRU_s017 >= 1
 z_k10_s0172_constraint:         var_10 + z_k10_s0172 + 1.3507 w_k10_HSBC_s017
                                 + 1.09976 w_k10_VOD_s017
                                 + 1.46603 w_k10_BP_s017 + 1.136 w_k10_GSK_s017
                                 + 1.01493 w_k10_AZN_s017
                                 + 1.42505 w_k10_RIO_s017
                                 + 1.06409 w_k10_BG_s017
                                 + 1.42888 w_k10_TSCO_s017
                                 + 0.957385 w_k10_AAL_s017
                                 + 1.07984 w_k10_PRU_s017 >= 1
 z_k10_s0173_constraint:         var_10 + z_k10_s0173 + 0.92317 w_k10_HSBC_s017
                                 + 0.8277 w_k10_VOD_s017
                                 + 0.779713 w_k10_BP_s017
                                 + 0.784455 w_k10_GSK_s017
                                 + 0.92825 w_k10_AZN_s017
                                 + 0.867058 w_k10_RIO_s017
                                 + 0.909896 w_k10_BG_s017
                                 + 0.871161 w_k10_TSCO_s017
                                 + 1.03108 w_k10_AAL_s017
                                 + 0.967135 w_k10_PRU_s017 >= 1
 z_k10_s0174_constraint:         var_10 + z_k10_s0174 + 1.0478 w_k10_HSBC_s017
                                 + 1.01417 w_k10_VOD_s017
                                 + 1.22651 w_k10_BP_s017
                                 + 1.08494 w_k10_GSK_s017
                                 + 1.03359 w_k10_AZN_s017
                                 + 1.08242 w_k10_RIO_s017
                                 + 0.953145 w_k10_BG_s017
                                 + 1.15923 w_k10_TSCO_s017
                                 + 1.0397 w_k10_AAL_s017
                                 + 1.01492 w_k10_PRU_s017 >= 1
 z_k10_s0175_constraint:         var_10 + z_k10_s0175
                                 + 0.954244 w_k10_HSBC_s017
                                 + 0.89132 w_k10_VOD_s017
                                 + 1.11131 w_k10_BP_s017
                                 + 1.02308 w_k10_GSK_s017
                                 + 0.985357 w_k10_AZN_s017
                                 + 1.02488 w_k10_RIO_s017
                                 + 0.881825 w_k10_BG_s017
                                 + 1.06889 w_k10_TSCO_s017
                                 + 0.964028 w_k10_AAL_s017
                                 + 1.30657 w_k10_PRU_s017 >= 1
 z_k10_s0176_constraint:         var_10 + z_k10_s0176
                                 + 0.998767 w_k10_HSBC_s017
                                 + 0.836529 w_k10_VOD_s017
                                 + 0.987516 w_k10_BP_s017
                                 + 0.968574 w_k10_GSK_s017
                                 + 1.02424 w_k10_AZN_s017
                                 + 0.913585 w_k10_RIO_s017
                                 + 1.00898 w_k10_BG_s017
                                 + 1.00829 w_k10_TSCO_s017
                                 + 0.98307 w_k10_AAL_s017
                                 + 0.854943 w_k10_PRU_s017 >= 1
 z_k10_s0177_constraint:         var_10 + z_k10_s0177 + 1.06372 w_k10_HSBC_s017
                                 + 1.01575 w_k10_VOD_s017
                                 + 1.12144 w_k10_BP_s017
                                 + 1.01342 w_k10_GSK_s017
                                 + 1.03409 w_k10_AZN_s017
                                 + 1.09131 w_k10_RIO_s017
                                 + 0.989173 w_k10_BG_s017
                                 + 1.37561 w_k10_TSCO_s017
                                 + 1.09714 w_k10_AAL_s017
                                 + 1.11029 w_k10_PRU_s017 >= 1
 z_k10_s1000_constraint:         var_10 + z_k10_s1000
                                 + 0.820757 w_k10_HSBC_s100
                                 + 1.02448 w_k10_VOD_s100
                                 + 0.882588 w_k10_BP_s100
                                 + 0.852181 w_k10_GSK_s100
                                 + 0.994876 w_k10_AZN_s100
                                 + 0.862456 w_k10_RIO_s100
                                 + 0.885098 w_k10_BG_s100
                                 + 0.813742 w_k10_TSCO_s100
                                 + 1.08287 w_k10_AAL_s100
                                 + 0.994027 w_k10_PRU_s100 >= 1
 z_k10_s1001_constraint:         var_10 + z_k10_s1001
                                 + 0.956108 w_k10_HSBC_s100
                                 + 1.18096 w_k10_VOD_s100
                                 + 1.1985 w_k10_BP_s100
                                 + 1.21158 w_k10_GSK_s100
                                 + 1.22815 w_k10_AZN_s100
                                 + 1.06552 w_k10_RIO_s100
                                 + 0.939532 w_k10_BG_s100
                                 + 1.29661 w_k10_TSCO_s100
                                 + 1.3273 w_k10_AAL_s100
                                 + 1.22278 w_k10_PRU_s100 >= 1
 z_k10_s1002_constraint:         var_10 + z_k10_s1002 + 1.12332 w_k10_HSBC_s100
                                 + 1.26353 w_k10_VOD_s100
                                 + 1.38522 w_k10_BP_s100
                                 + 1.19577 w_k10_GSK_s100
                                 + 1.13083 w_k10_AZN_s100
                                 + 1.2277 w_k10_RIO_s100
                                 + 0.936391 w_k10_BG_s100
                                 + 1.3893 w_k10_TSCO_s100
                                 + 1.05487 w_k10_AAL_s100
                                 + 1.28759 w_k10_PRU_s100 >= 1
 z_k10_s1003_constraint:         var_10 + z_k10_s1003
                                 + 0.951561 w_k10_HSBC_s100
                                 + 0.93998 w_k10_VOD_s100
                                 + 0.828389 w_k10_BP_s100
                                 + 0.852715 w_k10_GSK_s100
                                 + 0.858144 w_k10_AZN_s100
                                 + 0.870181 w_k10_RIO_s100
                                 + 1.02377 w_k10_BG_s100
                                 + 0.740121 w_k10_TSCO_s100
                                 + 0.948678 w_k10_AAL_s100
                                 + 0.679427 w_k10_PRU_s100 >= 1
 z_k10_s1004_constraint:         var_10 + z_k10_s1004 + 0.97671 w_k10_HSBC_s100
                                 + 0.978392 w_k10_VOD_s100
                                 + 1.01953 w_k10_BP_s100
                                 + 1.07638 w_k10_GSK_s100
                                 + 1.1733 w_k10_AZN_s100
                                 + 1.03235 w_k10_RIO_s100
                                 + 1.14116 w_k10_BG_s100
                                 + 1.20192 w_k10_TSCO_s100
                                 + 0.942341 w_k10_AAL_s100
                                 + 0.85987 w_k10_PRU_s100 >= 1
 z_k10_s1005_constraint:         var_10 + z_k10_s1005 + 1.07868 w_k10_HSBC_s100
                                 + 0.934836 w_k10_VOD_s100
                                 + 0.847183 w_k10_BP_s100
                                 + 0.934805 w_k10_GSK_s100
                                 + 1.05335 w_k10_AZN_s100
                                 + 0.927617 w_k10_RIO_s100
                                 + 0.914097 w_k10_BG_s100
                                 + 1.14002 w_k10_TSCO_s100
                                 + 0.98013 w_k10_AAL_s100
                                 + 0.885702 w_k10_PRU_s100 >= 1
 z_k10_s1006_constraint:         var_10 + z_k10_s1006 + 1.01495 w_k10_HSBC_s100
                                 + 1.10702 w_k10_VOD_s100
                                 + 1.06953 w_k10_BP_s100
                                 + 0.997897 w_k10_GSK_s100
                                 + 1.09072 w_k10_AZN_s100
                                 + 1.0359 w_k10_RIO_s100
                                 + 1.09055 w_k10_BG_s100
                                 + 1.16815 w_k10_TSCO_s100
                                 + 1.00015 w_k10_AAL_s100
                                 + 1.01032 w_k10_PRU_s100 >= 1
 z_k10_s1007_constraint:         var_10 + z_k10_s1007
                                 + 0.923508 w_k10_HSBC_s100
                                 + 1.05609 w_k10_VOD_s100
                                 + 1.08873 w_k10_BP_s100
                                 + 0.90822 w_k10_GSK_s100
                                 + 1.04974 w_k10_AZN_s100
                                 + 0.947042 w_k10_RIO_s100
                                 + 0.971381 w_k10_BG_s100
                                 + 1.09122 w_k10_TSCO_s100
                                 + 1.14088 w_k10_AAL_s100
                                 + 1.05297 w_k10_PRU_s100 >= 1
 z_k10_s1010_constraint:         var_10 + z_k10_s1010
                                 + 0.992594 w_k10_HSBC_s101
                                 + 1.07848 w_k10_VOD_s101
                                 + 1.00228 w_k10_BP_s101
                                 + 0.850085 w_k10_GSK_s101
                                 + 1.02415 w_k10_AZN_s101
                                 + 0.977665 w_k10_RIO_s101
                                 + 1.02554 w_k10_BG_s101
                                 + 1.17912 w_k10_TSCO_s101
                                 + 1.17967 w_k10_AAL_s101
                                 + 1.26774 w_k10_PRU_s101 >= 1
 z_k10_s1011_constraint:         var_10 + z_k10_s1011
                                 + 0.793627 w_k10_HSBC_s101
                                 + 0.886374 w_k10_VOD_s101
                                 + 0.897269 w_k10_BP_s101
                                 + 0.878767 w_k10_GSK_s101
                                 + 0.880043 w_k10_AZN_s101
                                 + 0.92973 w_k10_RIO_s101
                                 + 1.05733 w_k10_BG_s101
                                 + 0.945251 w_k10_TSCO_s101
                                 + 1.08897 w_k10_AAL_s101
                                 + 1.12724 w_k10_PRU_s101 >= 1
 z_k10_s1012_constraint:         var_10 + z_k10_s1012 + 1.15476 w_k10_HSBC_s101
                                 + 1.05087 w_k10_VOD_s101
                                 + 1.1749 w_k10_BP_s101
                                 + 1.03317 w_k10_GSK_s101
                                 + 1.05045 w_k10_AZN_s101
                                 + 1.26872 w_k10_RIO_s101
                                 + 1.08182 w_k10_BG_s101
                                 + 1.19023 w_k10_TSCO_s101
                                 + 0.833543 w_k10_AAL_s101
                                 + 1.184 w_k10_PRU_s101 >= 1
 z_k10_s1013_constraint:         var_10 + z_k10_s1013
                                 + 0.841274 w_k10_HSBC_s101
                                 + 1.02978 w_k10_VOD_s101
                                 + 0.861929 w_k10_BP_s101
                                 + 0.88742 w_k10_GSK_s101
                                 + 0.959231 w_k10_AZN_s101
                                 + 0.847644 w_k10_RIO_s101
                                 + 0.971454 w_k10_BG_s101
                                 + 0.775322 w_k10_TSCO_s101
                                 + 1.2483 w_k10_AAL_s101
                                 + 1.12025 w_k10_PRU_s101 >= 1
 z_k10_s1014_constraint:         var_10 + z_k10_s1014
                                 + 0.980821 w_k10_HSBC_s101
                                 + 1.11574 w_k10_VOD_s101
                                 + 1.30067 w_k10_BP_s101
                                 + 1.05148 w_k10_GSK_s101
                                 + 1.076 w_k10_AZN_s101
                                 + 1.17921 w_k10_RIO_s101
                                 + 0.906409 w_k10_BG_s101
                                 + 1.16487 w_k10_TSCO_s101
                                 + 1.06652 w_k10_AAL_s101
                                 + 1.51367 w_k10_PRU_s101 >= 1
 z_k10_s1015_constraint:         var_10 + z_k10_s1015 + 1.0398 w_k10_HSBC_s101
                                 + 0.963176 w_k10_VOD_s101
                                 + 0.919896 w_k10_BP_s101
                                 + 0.907239 w_k10_GSK_s101
                                 + 0.887483 w_k10_AZN_s101
                                 + 1.10602 w_k10_RIO_s101
                                 + 1.0971 w_k10_BG_s101
                                 + 1.10218 w_k10_TSCO_s101
                                 + 0.887953 w_k10_AAL_s101
                                 + 1.02944 w_k10_PRU_s101 >= 1
 z_k10_s1016_constraint:         var_10 + z_k10_s1016
                                 + 0.852748 w_k10_HSBC_s101
                                 + 0.95233 w_k10_VOD_s101
                                 + 0.689823 w_k10_BP_s101
                                 + 0.777726 w_k10_GSK_s101
                                 + 0.936846 w_k10_AZN_s101
                                 + 0.816122 w_k10_RIO_s101
                                 + 0.883876 w_k10_BG_s101
                                 + 0.762687 w_k10_TSCO_s101
                                 + 0.99322 w_k10_AAL_s101
                                 + 0.939865 w_k10_PRU_s101 >= 1
 z_k10_s1017_constraint:         var_10 + z_k10_s1017 + 1.15318 w_k10_HSBC_s101
                                 + 1.09274 w_k10_VOD_s101
                                 + 1.39782 w_k10_BP_s101
                                 + 1.09126 w_k10_GSK_s101
                                 + 1.0345 w_k10_AZN_s101
                                 + 1.32458 w_k10_RIO_s101
                                 + 1.11664 w_k10_BG_s101
                                 + 1.41506 w_k10_TSCO_s101
                                 + 1.05945 w_k10_AAL_s101
                                 + 1.28428 w_k10_PRU_s101 >= 1
 z_k10_s1020_constraint:         var_10 + z_k10_s1020 + 1.14737 w_k10_HSBC_s102
                                 + 1.01395 w_k10_VOD_s102
                                 + 1.19684 w_k10_BP_s102
                                 + 1.14337 w_k10_GSK_s102
                                 + 0.962501 w_k10_AZN_s102
                                 + 1.2511 w_k10_RIO_s102
                                 + 1.09567 w_k10_BG_s102
                                 + 1.28995 w_k10_TSCO_s102
                                 + 0.94136 w_k10_AAL_s102
                                 + 1.14347 w_k10_PRU_s102 >= 1
 z_k10_s1021_constraint:         var_10 + z_k10_s1021 + 1.30497 w_k10_HSBC_s102
                                 + 1.31257 w_k10_VOD_s102
                                 + 1.39857 w_k10_BP_s102
                                 + 1.22744 w_k10_GSK_s102
                                 + 1.19916 w_k10_AZN_s102
                                 + 1.48848 w_k10_RIO_s102
                                 + 1.08647 w_k10_BG_s102
                                 + 1.73721 w_k10_TSCO_s102
                                 + 0.91092 w_k10_AAL_s102
                                 + 1.10072 w_k10_PRU_s102 >= 1
 z_k10_s1022_constraint:         var_10 + z_k10_s1022 + 1.06836 w_k10_HSBC_s102
                                 + 1.11491 w_k10_VOD_s102
                                 + 1.03304 w_k10_BP_s102
                                 + 1.02857 w_k10_GSK_s102
                                 + 1.06297 w_k10_AZN_s102
                                 + 1.17955 w_k10_RIO_s102
                                 + 0.931194 w_k10_BG_s102
                                 + 0.993416 w_k10_TSCO_s102
                                 + 0.83907 w_k10_AAL_s102
                                 + 1.11489 w_k10_PRU_s102 >= 1
 z_k10_s1023_constraint:         var_10 + z_k10_s1023
                                 + 0.996966 w_k10_HSBC_s102
                                 + 1.20756 w_k10_VOD_s102
                                 + 1.08405 w_k10_BP_s102
                                 + 1.04106 w_k10_GSK_s102
                                 + 1.09881 w_k10_AZN_s102
                                 + 1.08902 w_k10_RIO_s102
                                 + 0.877083 w_k10_BG_s102
                                 + 0.912584 w_k10_TSCO_s102
                                 + 1.07245 w_k10_AAL_s102
                                 + 1.18292 w_k10_PRU_s102 >= 1
 z_k10_s1024_constraint:         var_10 + z_k10_s1024
                                 + 0.883315 w_k10_HSBC_s102
                                 + 1.04817 w_k10_VOD_s102
                                 + 0.806299 w_k10_BP_s102
                                 + 0.794439 w_k10_GSK_s102
                                 + 0.912539 w_k10_AZN_s102
                                 + 0.854458 w_k10_RIO_s102
                                 + 0.987126 w_k10_BG_s102
                                 + 0.69251 w_k10_TSCO_s102
                                 + 1.06273 w_k10_AAL_s102
                                 + 1.02146 w_k10_PRU_s102 >= 1
 z_k10_s1025_constraint:         var_10 + z_k10_s1025 + 1.08488 w_k10_HSBC_s102
                                 + 0.998421 w_k10_VOD_s102
                                 + 0.915366 w_k10_BP_s102
                                 + 0.856422 w_k10_GSK_s102
                                 + 1.00032 w_k10_AZN_s102
                                 + 0.965334 w_k10_RIO_s102
                                 + 1.0529 w_k10_BG_s102
                                 + 0.99455 w_k10_TSCO_s102
                                 + 0.95199 w_k10_AAL_s102
                                 + 0.933693 w_k10_PRU_s102 >= 1
 z_k10_s1026_constraint:         var_10 + z_k10_s1026
                                 + 0.967569 w_k10_HSBC_s102
                                 + 1.03017 w_k10_VOD_s102
                                 + 0.908404 w_k10_BP_s102
                                 + 1.02038 w_k10_GSK_s102
                                 + 1.11044 w_k10_AZN_s102
                                 + 0.938329 w_k10_RIO_s102
                                 + 0.889857 w_k10_BG_s102
                                 + 0.867531 w_k10_TSCO_s102
                                 + 0.987234 w_k10_AAL_s102
                                 + 1.02808 w_k10_PRU_s102 >= 1
 z_k10_s1027_constraint:         var_10 + z_k10_s1027 + 1.09196 w_k10_HSBC_s102
                                 + 1.09148 w_k10_VOD_s102
                                 + 1.10418 w_k10_BP_s102
                                 + 1.06904 w_k10_GSK_s102
                                 + 1.11034 w_k10_AZN_s102
                                 + 1.17278 w_k10_RIO_s102
                                 + 1.01393 w_k10_BG_s102
                                 + 1.1075 w_k10_TSCO_s102
                                 + 0.957064 w_k10_AAL_s102
                                 + 1.1153 w_k10_PRU_s102 >= 1
 z_k10_s1030_constraint:         var_10 + z_k10_s1030 + 1.30321 w_k10_HSBC_s103
                                 + 0.998746 w_k10_VOD_s103
                                 + 1.09259 w_k10_BP_s103
                                 + 1.0208 w_k10_GSK_s103
                                 + 1.03401 w_k10_AZN_s103
                                 + 1.08488 w_k10_RIO_s103
                                 + 1.04003 w_k10_BG_s103
                                 + 1.2039 w_k10_TSCO_s103
                                 + 1.09496 w_k10_AAL_s103
                                 + 0.99857 w_k10_PRU_s103 >= 1
 z_k10_s1031_constraint:         var_10 + z_k10_s1031
                                 + 0.868204 w_k10_HSBC_s103
                                 + 1.09004 w_k10_VOD_s103
                                 + 0.88835 w_k10_BP_s103
                                 + 0.996755 w_k10_GSK_s103
                                 + 1.02198 w_k10_AZN_s103
                                 + 0.93957 w_k10_RIO_s103
                                 + 0.908283 w_k10_BG_s103
                                 + 0.875305 w_k10_TSCO_s103
                                 + 0.91612 w_k10_AAL_s103
                                 + 0.853867 w_k10_PRU_s103 >= 1
 z_k10_s1032_constraint:         var_10 + z_k10_s1032
                                 + 0.779224 w_k10_HSBC_s103
                                 + 1.05774 w_k10_VOD_s103
                                 + 1.08538 w_k10_BP_s103
                                 + 1.1048 w_k10_GSK_s103
                                 + 0.976305 w_k10_AZN_s103
                                 + 1.06511 w_k10_RIO_s103
                                 + 1.00217 w_k10_BG_s103
                                 + 0.84824 w_k10_TSCO_s103
                                 + 1.15607 w_k10_AAL_s103
                                 + 1.11443 w_k10_PRU_s103 >= 1
 z_k10_s1033_constraint:         var_10 + z_k10_s1033 + 1.21505 w_k10_HSBC_s103
                                 + 1.13952 w_k10_VOD_s103
                                 + 1.28669 w_k10_BP_s103
                                 + 1.20408 w_k10_GSK_s103
                                 + 1.06532 w_k10_AZN_s103
                                 + 1.23747 w_k10_RIO_s103
                                 + 1.06377 w_k10_BG_s103
                                 + 1.20368 w_k10_TSCO_s103
                                 + 0.962636 w_k10_AAL_s103
                                 + 1.01541 w_k10_PRU_s103 >= 1
 z_k10_s1034_constraint:         var_10 + z_k10_s1034
                                 + 0.996383 w_k10_HSBC_s103
                                 + 1.0123 w_k10_VOD_s103
                                 + 0.994088 w_k10_BP_s103
                                 + 1.08622 w_k10_GSK_s103
                                 + 1.03636 w_k10_AZN_s103
                                 + 1.0358 w_k10_RIO_s103
                                 + 1.16442 w_k10_BG_s103
                                 + 0.848663 w_k10_TSCO_s103
                                 + 0.875728 w_k10_AAL_s103
                                 + 0.817975 w_k10_PRU_s103 >= 1
 z_k10_s1035_constraint:         var_10 + z_k10_s1035 + 1.05052 w_k10_HSBC_s103
                                 + 1.43864 w_k10_VOD_s103
                                 + 1.1332 w_k10_BP_s103
                                 + 1.09215 w_k10_GSK_s103
                                 + 1.20644 w_k10_AZN_s103
                                 + 1.06215 w_k10_RIO_s103
                                 + 0.867521 w_k10_BG_s103
                                 + 0.902486 w_k10_TSCO_s103
                                 + 1.24365 w_k10_AAL_s103
                                 + 1.30878 w_k10_PRU_s103 >= 1
 z_k10_s1036_constraint:         var_10 + z_k10_s1036 + 1.0101 w_k10_HSBC_s103
                                 + 1.02269 w_k10_VOD_s103
                                 + 1.05992 w_k10_BP_s103
                                 + 1.13282 w_k10_GSK_s103
                                 + 0.982085 w_k10_AZN_s103
                                 + 1.03878 w_k10_RIO_s103
                                 + 1.17768 w_k10_BG_s103
                                 + 0.880052 w_k10_TSCO_s103
                                 + 1.13244 w_k10_AAL_s103
                                 + 0.905815 w_k10_PRU_s103 >= 1
 z_k10_s1037_constraint:         var_10 + z_k10_s1037
                                 + 0.839097 w_k10_HSBC_s103
                                 + 0.917692 w_k10_VOD_s103
                                 + 0.728245 w_k10_BP_s103
                                 + 0.954143 w_k10_GSK_s103
                                 + 1.05172 w_k10_AZN_s103
                                 + 0.777074 w_k10_RIO_s103
                                 + 1.00469 w_k10_BG_s103
                                 + 0.758868 w_k10_TSCO_s103
                                 + 1.14531 w_k10_AAL_s103
                                 + 0.80211 w_k10_PRU_s103 >= 1
 z_k10_s1040_constraint:         var_10 + z_k10_s1040 + 1.04557 w_k10_HSBC_s104
                                 + 1.33931 w_k10_VOD_s104
                                 + 0.946101 w_k10_BP_s104
                                 + 0.915818 w_k10_GSK_s104
                                 + 1.0186 w_k10_AZN_s104
                                 + 1.00799 w_k10_RIO_s104
                                 + 0.874788 w_k10_BG_s104
                                 + 0.85894 w_k10_TSCO_s104
                                 + 1.02403 w_k10_AAL_s104
                                 + 1.03685 w_k10_PRU_s104 >= 1
 z_k10_s1041_constraint:         var_10 + z_k10_s1041
                                 + 0.812107 w_k10_HSBC_s104
                                 + 0.871236 w_k10_VOD_s104
                                 + 0.73123 w_k10_BP_s104
                                 + 0.893662 w_k10_GSK_s104
                                 + 0.966802 w_k10_AZN_s104
                                 + 0.811053 w_k10_RIO_s104
                                 + 0.942065 w_k10_BG_s104
                                 + 0.748622 w_k10_TSCO_s104
                                 + 0.946169 w_k10_AAL_s104
                                 + 0.814582 w_k10_PRU_s104 >= 1
 z_k10_s1042_constraint:         var_10 + z_k10_s1042 + 1.35841 w_k10_HSBC_s104
                                 + 0.994068 w_k10_VOD_s104
                                 + 1.15106 w_k10_BP_s104
                                 + 1.0451 w_k10_GSK_s104
                                 + 0.971951 w_k10_AZN_s104
                                 + 1.17405 w_k10_RIO_s104
                                 + 1.06171 w_k10_BG_s104
                                 + 1.33877 w_k10_TSCO_s104
                                 + 0.931558 w_k10_AAL_s104
                                 + 0.885245 w_k10_PRU_s104 >= 1
 z_k10_s1043_constraint:         var_10 + z_k10_s1043 + 1.02848 w_k10_HSBC_s104
                                 + 1.03298 w_k10_VOD_s104
                                 + 0.956778 w_k10_BP_s104
                                 + 0.992633 w_k10_GSK_s104
                                 + 1.03935 w_k10_AZN_s104
                                 + 0.947087 w_k10_RIO_s104
                                 + 0.929557 w_k10_BG_s104
                                 + 1.14418 w_k10_TSCO_s104
                                 + 0.900007 w_k10_AAL_s104
                                 + 0.955451 w_k10_PRU_s104 >= 1
 z_k10_s1044_constraint:         var_10 + z_k10_s1044 + 1.0054 w_k10_HSBC_s104
                                 + 0.84139 w_k10_VOD_s104
                                 + 0.895008 w_k10_BP_s104
                                 + 0.805924 w_k10_GSK_s104
                                 + 0.916476 w_k10_AZN_s104
                                 + 0.848254 w_k10_RIO_s104
                                 + 0.93343 w_k10_BG_s104
                                 + 0.870632 w_k10_TSCO_s104
                                 + 1.0698 w_k10_AAL_s104
                                 + 0.913504 w_k10_PRU_s104 >= 1
 z_k10_s1045_constraint:         var_10 + z_k10_s1045
                                 + 0.985592 w_k10_HSBC_s104
                                 + 0.968473 w_k10_VOD_s104
                                 + 1.10263 w_k10_BP_s104
                                 + 1.06716 w_k10_GSK_s104
                                 + 0.999593 w_k10_AZN_s104
                                 + 1.05129 w_k10_RIO_s104
                                 + 1.12084 w_k10_BG_s104
                                 + 1.16632 w_k10_TSCO_s104
                                 + 0.949037 w_k10_AAL_s104
                                 + 0.695863 w_k10_PRU_s104 >= 1
 z_k10_s1046_constraint:         var_10 + z_k10_s1046 + 1.20658 w_k10_HSBC_s104
                                 + 1.17021 w_k10_VOD_s104
                                 + 1.24803 w_k10_BP_s104
                                 + 1.11822 w_k10_GSK_s104
                                 + 1.0814 w_k10_AZN_s104
                                 + 1.23339 w_k10_RIO_s104
                                 + 1.0092 w_k10_BG_s104
                                 + 1.2864 w_k10_TSCO_s104
                                 + 0.853779 w_k10_AAL_s104
                                 + 1.00327 w_k10_PRU_s104 >= 1
 z_k10_s1047_constraint:         var_10 + z_k10_s1047 + 1.17619 w_k10_HSBC_s104
                                 + 1.20683 w_k10_VOD_s104
                                 + 1.12375 w_k10_BP_s104
                                 + 1.03651 w_k10_GSK_s104
                                 + 1.11137 w_k10_AZN_s104
                                 + 1.22483 w_k10_RIO_s104
                                 + 1.0278 w_k10_BG_s104
                                 + 1.13949 w_k10_TSCO_s104
                                 + 1.24281 w_k10_AAL_s104
                                 + 1.04693 w_k10_PRU_s104 >= 1
 z_k10_s1050_constraint:         var_10 + z_k10_s1050 + 1.05712 w_k10_HSBC_s105
                                 + 0.877298 w_k10_VOD_s105
                                 + 0.996047 w_k10_BP_s105
                                 + 0.973053 w_k10_GSK_s105
                                 + 1.04365 w_k10_AZN_s105
                                 + 0.937478 w_k10_RIO_s105
                                 + 1.04529 w_k10_BG_s105
                                 + 1.07947 w_k10_TSCO_s105
                                 + 0.936097 w_k10_AAL_s105
                                 + 1.05434 w_k10_PRU_s105 >= 1
 z_k10_s1051_constraint:         var_10 + z_k10_s1051 + 1.00021 w_k10_HSBC_s105
                                 + 1.09586 w_k10_VOD_s105
                                 + 1.28103 w_k10_BP_s105
                                 + 1.17801 w_k10_GSK_s105
                                 + 1.17239 w_k10_AZN_s105
                                 + 1.24466 w_k10_RIO_s105
                                 + 1.04991 w_k10_BG_s105
                                 + 1.37691 w_k10_TSCO_s105
                                 + 0.983853 w_k10_AAL_s105
                                 + 1.10676 w_k10_PRU_s105 >= 1
 z_k10_s1052_constraint:         var_10 + z_k10_s1052
                                 + 0.817495 w_k10_HSBC_s105
                                 + 0.838813 w_k10_VOD_s105
                                 + 0.937448 w_k10_BP_s105
                                 + 0.997633 w_k10_GSK_s105
                                 + 1.00942 w_k10_AZN_s105
                                 + 0.923805 w_k10_RIO_s105
                                 + 0.885465 w_k10_BG_s105
                                 + 0.788974 w_k10_TSCO_s105
                                 + 1.02836 w_k10_AAL_s105
                                 + 1.28582 w_k10_PRU_s105 >= 1
 z_k10_s1053_constraint:         var_10 + z_k10_s1053
                                 + 0.841881 w_k10_HSBC_s105
                                 + 0.958033 w_k10_VOD_s105
                                 + 0.796956 w_k10_BP_s105
                                 + 0.88292 w_k10_GSK_s105
                                 + 1.01863 w_k10_AZN_s105
                                 + 0.821369 w_k10_RIO_s105
                                 + 0.859793 w_k10_BG_s105
                                 + 0.757598 w_k10_TSCO_s105
                                 + 1.00083 w_k10_AAL_s105
                                 + 1.24999 w_k10_PRU_s105 >= 1
 z_k10_s1054_constraint:         var_10 + z_k10_s1054
                                 + 0.821189 w_k10_HSBC_s105
                                 + 0.841411 w_k10_VOD_s105
                                 + 0.860962 w_k10_BP_s105
                                 + 0.942534 w_k10_GSK_s105
                                 + 0.958278 w_k10_AZN_s105
                                 + 0.89651 w_k10_RIO_s105
                                 + 1.0092 w_k10_BG_s105
                                 + 0.829121 w_k10_TSCO_s105
                                 + 0.978556 w_k10_AAL_s105
                                 + 0.941823 w_k10_PRU_s105 >= 1
 z_k10_s1055_constraint:         var_10 + z_k10_s1055 + 1.09692 w_k10_HSBC_s105
                                 + 0.872638 w_k10_VOD_s105
                                 + 0.811115 w_k10_BP_s105
                                 + 0.980796 w_k10_GSK_s105
                                 + 1.02175 w_k10_AZN_s105
                                 + 0.883631 w_k10_RIO_s105
                                 + 0.935052 w_k10_BG_s105
                                 + 0.88554 w_k10_TSCO_s105
                                 + 0.877526 w_k10_AAL_s105
                                 + 0.953793 w_k10_PRU_s105 >= 1
 z_k10_s1056_constraint:         var_10 + z_k10_s1056
                                 + 0.804275 w_k10_HSBC_s105
                                 + 0.766283 w_k10_VOD_s105
                                 + 0.917472 w_k10_BP_s105
                                 + 1.02596 w_k10_GSK_s105
                                 + 1.00271 w_k10_AZN_s105
                                 + 0.952167 w_k10_RIO_s105
                                 + 1.19414 w_k10_BG_s105
                                 + 0.945761 w_k10_TSCO_s105
                                 + 0.80983 w_k10_AAL_s105
                                 + 0.993833 w_k10_PRU_s105 >= 1
 z_k10_s1057_constraint:         var_10 + z_k10_s1057
                                 + 0.927026 w_k10_HSBC_s105
                                 + 1.00749 w_k10_VOD_s105
                                 + 1.12485 w_k10_BP_s105
                                 + 1.04816 w_k10_GSK_s105
                                 + 1.01194 w_k10_AZN_s105
                                 + 1.09689 w_k10_RIO_s105
                                 + 0.978077 w_k10_BG_s105
                                 + 0.995947 w_k10_TSCO_s105
                                 + 1.1706 w_k10_AAL_s105
                                 + 1.24691 w_k10_PRU_s105 >= 1
 z_k10_s1060_constraint:         var_10 + z_k10_s1060
                                 + 0.814566 w_k10_HSBC_s106
                                 + 1.06928 w_k10_VOD_s106
                                 + 1.00335 w_k10_BP_s106
                                 + 1.00203 w_k10_GSK_s106
                                 + 0.99124 w_k10_AZN_s106
                                 + 0.940809 w_k10_RIO_s106
                                 + 0.924265 w_k10_BG_s106
                                 + 0.884148 w_k10_TSCO_s106
                                 + 1.14211 w_k10_AAL_s106
                                 + 1.09207 w_k10_PRU_s106 >= 1
 z_k10_s1061_constraint:         var_10 + z_k10_s1061 + 1.06754 w_k10_HSBC_s106
                                 + 1.19078 w_k10_VOD_s106
                                 + 1.15732 w_k10_BP_s106
                                 + 1.08912 w_k10_GSK_s106
                                 + 0.96054 w_k10_AZN_s106
                                 + 1.02659 w_k10_RIO_s106
                                 + 0.990613 w_k10_BG_s106
                                 + 1.10527 w_k10_TSCO_s106
                                 + 0.89763 w_k10_AAL_s106
                                 + 0.873396 w_k10_PRU_s106 >= 1
 z_k10_s1062_constraint:         var_10 + z_k10_s1062
                                 + 0.988487 w_k10_HSBC_s106
                                 + 1.02214 w_k10_VOD_s106
                                 + 1.03432 w_k10_BP_s106
                                 + 1.01445 w_k10_GSK_s106
                                 + 0.986003 w_k10_AZN_s106
                                 + 1.07879 w_k10_RIO_s106
                                 + 1.12755 w_k10_BG_s106
                                 + 1.08093 w_k10_TSCO_s106
                                 + 0.902395 w_k10_AAL_s106
                                 + 0.89363 w_k10_PRU_s106 >= 1
 z_k10_s1063_constraint:         var_10 + z_k10_s1063 + 1.17179 w_k10_HSBC_s106
                                 + 1.15004 w_k10_VOD_s106
                                 + 1.28514 w_k10_BP_s106
                                 + 1.08904 w_k10_GSK_s106
                                 + 0.980199 w_k10_AZN_s106
                                 + 1.22392 w_k10_RIO_s106
                                 + 1.15009 w_k10_BG_s106
                                 + 1.47912 w_k10_TSCO_s106
                                 + 0.867687 w_k10_AAL_s106
                                 + 0.976067 w_k10_PRU_s106 >= 1
 z_k10_s1064_constraint:         var_10 + z_k10_s1064 + 0.95982 w_k10_HSBC_s106
                                 + 1.14907 w_k10_VOD_s106
                                 + 1.25338 w_k10_BP_s106
                                 + 1.20888 w_k10_GSK_s106
                                 + 1.14607 w_k10_AZN_s106
                                 + 1.06647 w_k10_RIO_s106
                                 + 0.957709 w_k10_BG_s106
                                 + 1.28753 w_k10_TSCO_s106
                                 + 1.08643 w_k10_AAL_s106
                                 + 1.12632 w_k10_PRU_s106 >= 1
 z_k10_s1065_constraint:         var_10 + z_k10_s1065
                                 + 0.880151 w_k10_HSBC_s106
                                 + 0.956093 w_k10_VOD_s106
                                 + 0.889598 w_k10_BP_s106
                                 + 0.926424 w_k10_GSK_s106
                                 + 0.943911 w_k10_AZN_s106
                                 + 0.914999 w_k10_RIO_s106
                                 + 0.916173 w_k10_BG_s106
                                 + 0.816807 w_k10_TSCO_s106
                                 + 0.990732 w_k10_AAL_s106
                                 + 0.834564 w_k10_PRU_s106 >= 1
 z_k10_s1066_constraint:         var_10 + z_k10_s1066
                                 + 0.785709 w_k10_HSBC_s106
                                 + 1.02193 w_k10_VOD_s106
                                 + 0.862484 w_k10_BP_s106
                                 + 0.941834 w_k10_GSK_s106
                                 + 0.88651 w_k10_AZN_s106
                                 + 0.935872 w_k10_RIO_s106
                                 + 0.976475 w_k10_BG_s106
                                 + 0.870855 w_k10_TSCO_s106
                                 + 0.868253 w_k10_AAL_s106
                                 + 0.864443 w_k10_PRU_s106 >= 1
 z_k10_s1067_constraint:         var_10 + z_k10_s1067 + 1.04342 w_k10_HSBC_s106
                                 + 0.868373 w_k10_VOD_s106
                                 + 1.01545 w_k10_BP_s106
                                 + 0.9298 w_k10_GSK_s106
                                 + 0.926209 w_k10_AZN_s106
                                 + 0.84179 w_k10_RIO_s106
                                 + 0.994847 w_k10_BG_s106
                                 + 1.10992 w_k10_TSCO_s106
                                 + 1.05899 w_k10_AAL_s106
                                 + 0.921985 w_k10_PRU_s106 >= 1
 z_k10_s1070_constraint:         var_10 + z_k10_s1070
                                 + 0.974204 w_k10_HSBC_s107
                                 + 0.898024 w_k10_VOD_s107
                                 + 0.848001 w_k10_BP_s107
                                 + 0.825592 w_k10_GSK_s107
                                 + 0.989321 w_k10_AZN_s107
                                 + 0.867982 w_k10_RIO_s107
                                 + 1.08213 w_k10_BG_s107
                                 + 0.781019 w_k10_TSCO_s107
                                 + 0.870347 w_k10_AAL_s107
                                 + 0.997217 w_k10_PRU_s107 >= 1
 z_k10_s1071_constraint:         var_10 + z_k10_s1071 + 1.00439 w_k10_HSBC_s107
                                 + 1.02156 w_k10_VOD_s107
                                 + 1.08461 w_k10_BP_s107
                                 + 1.03371 w_k10_GSK_s107
                                 + 1.01201 w_k10_AZN_s107
                                 + 1.01965 w_k10_RIO_s107
                                 + 1.09649 w_k10_BG_s107
                                 + 1.04583 w_k10_TSCO_s107
                                 + 0.851155 w_k10_AAL_s107
                                 + 1.03977 w_k10_PRU_s107 >= 1
 z_k10_s1072_constraint:         var_10 + z_k10_s1072 + 1.44593 w_k10_HSBC_s107
                                 + 1.08751 w_k10_VOD_s107
                                 + 1.17635 w_k10_BP_s107
                                 + 1.1419 w_k10_GSK_s107
                                 + 1.08326 w_k10_AZN_s107
                                 + 1.21263 w_k10_RIO_s107
                                 + 1.17203 w_k10_BG_s107
                                 + 1.29334 w_k10_TSCO_s107
                                 + 0.780418 w_k10_AAL_s107
                                 + 1.01516 w_k10_PRU_s107 >= 1
 z_k10_s1073_constraint:         var_10 + z_k10_s1073 + 1.00813 w_k10_HSBC_s107
                                 + 0.682775 w_k10_VOD_s107
                                 + 0.943234 w_k10_BP_s107
                                 + 1.03821 w_k10_GSK_s107
                                 + 0.848585 w_k10_AZN_s107
                                 + 0.946264 w_k10_RIO_s107
                                 + 1.24739 w_k10_BG_s107
                                 + 1.0306 w_k10_TSCO_s107
                                 + 0.981061 w_k10_AAL_s107
                                 + 0.850439 w_k10_PRU_s107 >= 1
 z_k10_s1074_constraint:         var_10 + z_k10_s1074 + 1.18445 w_k10_HSBC_s107
                                 + 1.05263 w_k10_VOD_s107
                                 + 1.04527 w_k10_BP_s107
                                 + 1.10871 w_k10_GSK_s107
                                 + 1.12679 w_k10_AZN_s107
                                 + 1.07303 w_k10_RIO_s107
                                 + 1.03056 w_k10_BG_s107
                                 + 1.16413 w_k10_TSCO_s107
                                 + 1.11587 w_k10_AAL_s107
                                 + 1.18945 w_k10_PRU_s107 >= 1
 z_k10_s1075_constraint:         var_10 + z_k10_s1075 + 0.83896 w_k10_HSBC_s107
                                 + 1.16991 w_k10_VOD_s107
                                 + 1.37256 w_k10_BP_s107
                                 + 1.23082 w_k10_GSK_s107
                                 + 1.14658 w_k10_AZN_s107
                                 + 1.27455 w_k10_RIO_s107
                                 + 1.18639 w_k10_BG_s107
                                 + 1.15234 w_k10_TSCO_s107
                                 + 1.17029 w_k10_AAL_s107
                                 + 1.27864 w_k10_PRU_s107 >= 1
 z_k10_s1076_constraint:         var_10 + z_k10_s1076
                                 + 0.808363 w_k10_HSBC_s107
                                 + 0.913525 w_k10_VOD_s107
                                 + 0.85838 w_k10_BP_s107
                                 + 0.901161 w_k10_GSK_s107
                                 + 0.997822 w_k10_AZN_s107
                                 + 0.850298 w_k10_RIO_s107
                                 + 0.939565 w_k10_BG_s107
                                 + 0.768103 w_k10_TSCO_s107
                                 + 1.08304 w_k10_AAL_s107
                                 + 1.06297 w_k10_PRU_s107 >= 1
 z_k10_s1077_constraint:         var_10 + z_k10_s1077
                                 + 0.917561 w_k10_HSBC_s107
                                 + 1.07024 w_k10_VOD_s107
                                 + 0.900752 w_k10_BP_s107
                                 + 0.970074 w_k10_GSK_s107
                                 + 1.0199 w_k10_AZN_s107
                                 + 0.976387 w_k10_RIO_s107
                                 + 0.948085 w_k10_BG_s107
                                 + 0.931441 w_k10_TSCO_s107
                                 + 1.01355 w_k10_AAL_s107
                                 + 1.1345 w_k10_PRU_s107 >= 1
 z_k10_s1100_constraint:         var_10 + z_k10_s1100 + 1.00699 w_k10_HSBC_s110
                                 + 1.00473 w_k10_VOD_s110
                                 + 0.946701 w_k10_BP_s110
                                 + 0.905517 w_k10_GSK_s110
                                 + 0.863625 w_k10_AZN_s110
                                 + 0.917839 w_k10_RIO_s110
                                 + 0.916663 w_k10_BG_s110
                                 + 0.863671 w_k10_TSCO_s110
                                 + 1.03771 w_k10_AAL_s110
                                 + 1.04059 w_k10_PRU_s110 >= 1
 z_k10_s1101_constraint:         var_10 + z_k10_s1101
                                 + 0.993127 w_k10_HSBC_s110
                                 + 0.773165 w_k10_VOD_s110
                                 + 0.790376 w_k10_BP_s110
                                 + 0.678612 w_k10_GSK_s110
                                 + 0.769927 w_k10_AZN_s110
                                 + 0.700388 w_k10_RIO_s110
                                 + 0.915706 w_k10_BG_s110
                                 + 0.837083 w_k10_TSCO_s110
                                 + 0.949857 w_k10_AAL_s110
                                 + 0.818129 w_k10_PRU_s110 >= 1
 z_k10_s1102_constraint:         var_10 + z_k10_s1102 + 1.04593 w_k10_HSBC_s110
                                 + 1.0326 w_k10_VOD_s110
                                 + 1.08908 w_k10_BP_s110
                                 + 0.981521 w_k10_GSK_s110
                                 + 1.00749 w_k10_AZN_s110
                                 + 0.927138 w_k10_RIO_s110
                                 + 0.861464 w_k10_BG_s110
                                 + 1.09369 w_k10_TSCO_s110
                                 + 1.16166 w_k10_AAL_s110
                                 + 1.30969 w_k10_PRU_s110 >= 1
 z_k10_s1103_constraint:         var_10 + z_k10_s1103 + 1.11027 w_k10_HSBC_s110
                                 + 0.914076 w_k10_VOD_s110
                                 + 1.12424 w_k10_BP_s110
                                 + 1.00538 w_k10_GSK_s110
                                 + 0.85309 w_k10_AZN_s110
                                 + 0.952802 w_k10_RIO_s110
                                 + 1.0994 w_k10_BG_s110
                                 + 1.2162 w_k10_TSCO_s110
                                 + 0.974953 w_k10_AAL_s110
                                 + 0.820171 w_k10_PRU_s110 >= 1
 z_k10_s1104_constraint:         var_10 + z_k10_s1104 + 1.16379 w_k10_HSBC_s110
                                 + 0.860585 w_k10_VOD_s110
                                 + 0.960316 w_k10_BP_s110
                                 + 0.99826 w_k10_GSK_s110
                                 + 0.86604 w_k10_AZN_s110
                                 + 1.01475 w_k10_RIO_s110
                                 + 1.02624 w_k10_BG_s110
                                 + 0.966765 w_k10_TSCO_s110
                                 + 0.810982 w_k10_AAL_s110
                                 + 0.840646 w_k10_PRU_s110 >= 1
 z_k10_s1105_constraint:         var_10 + z_k10_s1105 + 1.31644 w_k10_HSBC_s110
                                 + 0.97464 w_k10_VOD_s110
                                 + 1.03045 w_k10_BP_s110
                                 + 1.04041 w_k10_GSK_s110
                                 + 0.963193 w_k10_AZN_s110
                                 + 1.00677 w_k10_RIO_s110
                                 + 1.05533 w_k10_BG_s110
                                 + 1.1832 w_k10_TSCO_s110
                                 + 0.948378 w_k10_AAL_s110
                                 + 0.986112 w_k10_PRU_s110 >= 1
 z_k10_s1106_constraint:         var_10 + z_k10_s1106
                                 + 0.875748 w_k10_HSBC_s110
                                 + 0.864745 w_k10_VOD_s110
                                 + 0.931868 w_k10_BP_s110
                                 + 0.879654 w_k10_GSK_s110
                                 + 0.773027 w_k10_AZN_s110
                                 + 0.826201 w_k10_RIO_s110
                                 + 1.12636 w_k10_BG_s110
                                 + 0.811749 w_k10_TSCO_s110
                                 + 0.874695 w_k10_AAL_s110
                                 + 0.761703 w_k10_PRU_s110 >= 1
 z_k10_s1107_constraint:         var_10 + z_k10_s1107 + 1.07049 w_k10_HSBC_s110
                                 + 1.11174 w_k10_VOD_s110
                                 + 1.28905 w_k10_BP_s110
                                 + 1.17006 w_k10_GSK_s110
                                 + 0.998856 w_k10_AZN_s110
                                 + 1.23032 w_k10_RIO_s110
                                 + 1.11362 w_k10_BG_s110
                                 + 1.29267 w_k10_TSCO_s110
                                 + 0.936406 w_k10_AAL_s110
                                 + 1.21137 w_k10_PRU_s110 >= 1
 z_k10_s1110_constraint:         var_10 + z_k10_s1110
                                 + 0.909767 w_k10_HSBC_s111
                                 + 0.834899 w_k10_VOD_s111
                                 + 0.933768 w_k10_BP_s111
                                 + 0.999165 w_k10_GSK_s111
                                 + 0.96651 w_k10_AZN_s111
                                 + 0.816498 w_k10_RIO_s111
                                 + 0.827195 w_k10_BG_s111
                                 + 0.935775 w_k10_TSCO_s111
                                 + 1.17625 w_k10_AAL_s111
                                 + 1.04014 w_k10_PRU_s111 >= 1
 z_k10_s1111_constraint:         var_10 + z_k10_s1111
                                 + 0.943734 w_k10_HSBC_s111
                                 + 0.883574 w_k10_VOD_s111
                                 + 1.09236 w_k10_BP_s111
                                 + 0.95257 w_k10_GSK_s111
                                 + 0.87344 w_k10_AZN_s111
                                 + 0.998816 w_k10_RIO_s111
                                 + 1.01405 w_k10_BG_s111
                                 + 1.08356 w_k10_TSCO_s111
                                 + 1.16419 w_k10_AAL_s111
                                 + 1.05615 w_k10_PRU_s111 >= 1
 z_k10_s1112_constraint:         var_10 + z_k10_s1112 + 1.03761 w_k10_HSBC_s111
                                 + 1.01113 w_k10_VOD_s111
                                 + 1.2384 w_k10_BP_s111
                                 + 1.02592 w_k10_GSK_s111
                                 + 0.92773 w_k10_AZN_s111
                                 + 1.09695 w_k10_RIO_s111
                                 + 0.869074 w_k10_BG_s111
                                 + 1.21282 w_k10_TSCO_s111
                                 + 1.09075 w_k10_AAL_s111
                                 + 1.21949 w_k10_PRU_s111 >= 1
 z_k10_s1113_constraint:         var_10 + z_k10_s1113 + 1.25573 w_k10_HSBC_s111
                                 + 1.12579 w_k10_VOD_s111
                                 + 1.47824 w_k10_BP_s111
                                 + 1.34814 w_k10_GSK_s111
                                 + 0.99543 w_k10_AZN_s111
                                 + 1.42908 w_k10_RIO_s111
                                 + 1.13303 w_k10_BG_s111
                                 + 1.48896 w_k10_TSCO_s111
                                 + 0.994887 w_k10_AAL_s111
                                 + 1.14717 w_k10_PRU_s111 >= 1
 z_k10_s1114_constraint:         var_10 + z_k10_s1114 + 1.15966 w_k10_HSBC_s111
                                 + 0.936729 w_k10_VOD_s111
                                 + 1.25513 w_k10_BP_s111
                                 + 1.13696 w_k10_GSK_s111
                                 + 1.02504 w_k10_AZN_s111
                                 + 1.06554 w_k10_RIO_s111
                                 + 1.12396 w_k10_BG_s111
                                 + 1.33854 w_k10_TSCO_s111
                                 + 0.949874 w_k10_AAL_s111
                                 + 0.925607 w_k10_PRU_s111 >= 1
 z_k10_s1115_constraint:         var_10 + z_k10_s1115 + 1.0848 w_k10_HSBC_s111
                                 + 0.663341 w_k10_VOD_s111
                                 + 0.888297 w_k10_BP_s111
                                 + 0.892878 w_k10_GSK_s111
                                 + 0.809249 w_k10_AZN_s111
                                 + 0.925186 w_k10_RIO_s111
                                 + 1.20799 w_k10_BG_s111
                                 + 0.901915 w_k10_TSCO_s111
                                 + 0.864449 w_k10_AAL_s111
                                 + 0.698739 w_k10_PRU_s111 >= 1
 z_k10_s1116_constraint:         var_10 + z_k10_s1116
                                 + 0.761835 w_k10_HSBC_s111
                                 + 0.831535 w_k10_VOD_s111
                                 + 0.729864 w_k10_BP_s111
                                 + 0.772036 w_k10_GSK_s111
                                 + 0.814635 w_k10_AZN_s111
                                 + 0.754902 w_k10_RIO_s111
                                 + 0.926625 w_k10_BG_s111
                                 + 0.671153 w_k10_TSCO_s111
                                 + 1.00712 w_k10_AAL_s111
                                 + 0.858966 w_k10_PRU_s111 >= 1
 z_k10_s1117_constraint:         var_10 + z_k10_s1117
                                 + 0.976943 w_k10_HSBC_s111
                                 + 0.93874 w_k10_VOD_s111
                                 + 1.00562 w_k10_BP_s111
                                 + 0.932309 w_k10_GSK_s111
                                 + 0.888182 w_k10_AZN_s111
                                 + 0.976509 w_k10_RIO_s111
                                 + 0.991404 w_k10_BG_s111
                                 + 1.02759 w_k10_TSCO_s111
                                 + 0.949782 w_k10_AAL_s111
                                 + 0.837786 w_k10_PRU_s111 >= 1
 z_k10_s1120_constraint:         var_10 + z_k10_s1120 + 1.0055 w_k10_HSBC_s112
                                 + 0.858964 w_k10_VOD_s112
                                 + 0.769333 w_k10_BP_s112
                                 + 0.794864 w_k10_GSK_s112
                                 + 0.885805 w_k10_AZN_s112
                                 + 0.812421 w_k10_RIO_s112
                                 + 0.844717 w_k10_BG_s112
                                 + 0.710508 w_k10_TSCO_s112
                                 + 0.941805 w_k10_AAL_s112
                                 + 0.975842 w_k10_PRU_s112 >= 1
 z_k10_s1121_constraint:         var_10 + z_k10_s1121 + 1.21183 w_k10_HSBC_s112
                                 + 0.831153 w_k10_VOD_s112
                                 + 1.0765 w_k10_BP_s112 + 0.9952 w_k10_GSK_s112
                                 + 0.822548 w_k10_AZN_s112
                                 + 1.28376 w_k10_RIO_s112
                                 + 1.21557 w_k10_BG_s112
                                 + 1.2772 w_k10_TSCO_s112
                                 + 1.03583 w_k10_AAL_s112
                                 + 1.08981 w_k10_PRU_s112 >= 1
 z_k10_s1122_constraint:         var_10 + z_k10_s1122
                                 + 0.983834 w_k10_HSBC_s112
                                 + 0.987016 w_k10_VOD_s112
                                 + 1.00549 w_k10_BP_s112
                                 + 1.03853 w_k10_GSK_s112
                                 + 1.10749 w_k10_AZN_s112
                                 + 1.00317 w_k10_RIO_s112
                                 + 0.930858 w_k10_BG_s112
                                 + 1.04818 w_k10_TSCO_s112
                                 + 0.987624 w_k10_AAL_s112
                                 + 1.16753 w_k10_PRU_s112 >= 1
 z_k10_s1123_constraint:         var_10 + z_k10_s1123 + 1.12284 w_k10_HSBC_s112
                                 + 0.920734 w_k10_VOD_s112
                                 + 0.987774 w_k10_BP_s112
                                 + 0.902336 w_k10_GSK_s112
                                 + 0.963511 w_k10_AZN_s112
                                 + 1.079 w_k10_RIO_s112 + 1.16436 w_k10_BG_s112
                                 + 0.963547 w_k10_TSCO_s112
                                 + 0.924642 w_k10_AAL_s112
                                 + 0.880552 w_k10_PRU_s112 >= 1
 z_k10_s1124_constraint:         var_10 + z_k10_s1124 + 1.19639 w_k10_HSBC_s112
                                 + 0.999679 w_k10_VOD_s112
                                 + 1.26268 w_k10_BP_s112
                                 + 1.0809 w_k10_GSK_s112
                                 + 0.91496 w_k10_AZN_s112
                                 + 1.28113 w_k10_RIO_s112
                                 + 0.962268 w_k10_BG_s112
                                 + 1.44674 w_k10_TSCO_s112
                                 + 1.03786 w_k10_AAL_s112
                                 + 1.29928 w_k10_PRU_s112 >= 1
 z_k10_s1125_constraint:         var_10 + z_k10_s1125 + 1.10099 w_k10_HSBC_s112
                                 + 0.942183 w_k10_VOD_s112
                                 + 0.988212 w_k10_BP_s112
                                 + 1.02189 w_k10_GSK_s112
                                 + 0.952519 w_k10_AZN_s112
                                 + 1.11052 w_k10_RIO_s112
                                 + 0.927707 w_k10_BG_s112
                                 + 1.07687 w_k10_TSCO_s112
                                 + 1.05199 w_k10_AAL_s112
                                 + 1.27529 w_k10_PRU_s112 >= 1
 z_k10_s1126_constraint:         var_10 + z_k10_s1126
                                 + 0.825574 w_k10_HSBC_s112
                                 + 0.942257 w_k10_VOD_s112
                                 + 0.884926 w_k10_BP_s112
                                 + 0.935565 w_k10_GSK_s112
                                 + 0.977269 w_k10_AZN_s112
                                 + 0.933502 w_k10_RIO_s112
                                 + 0.969258 w_k10_BG_s112
                                 + 0.842829 w_k10_TSCO_s112
                                 + 1.02574 w_k10_AAL_s112
                                 + 0.948473 w_k10_PRU_s112 >= 1
 z_k10_s1127_constraint:         var_10 + z_k10_s1127 + 1.40863 w_k10_HSBC_s112
                                 + 0.851974 w_k10_VOD_s112
                                 + 1.08433 w_k10_BP_s112
                                 + 1.04274 w_k10_GSK_s112
                                 + 0.852067 w_k10_AZN_s112
                                 + 1.15127 w_k10_RIO_s112
                                 + 1.05482 w_k10_BG_s112
                                 + 1.04637 w_k10_TSCO_s112
                                 + 0.943298 w_k10_AAL_s112
                                 + 1.02308 w_k10_PRU_s112 >= 1
 z_k10_s1130_constraint:         var_10 + z_k10_s1130
                                 + 0.764166 w_k10_HSBC_s113
                                 + 1.09498 w_k10_VOD_s113
                                 + 1.04443 w_k10_BP_s113
                                 + 0.970108 w_k10_GSK_s113
                                 + 1.01893 w_k10_AZN_s113
                                 + 0.947486 w_k10_RIO_s113
                                 + 0.920625 w_k10_BG_s113
                                 + 0.78504 w_k10_TSCO_s113
                                 + 1.13232 w_k10_AAL_s113
                                 + 1.0671 w_k10_PRU_s113 >= 1
 z_k10_s1131_constraint:         var_10 + z_k10_s1131
                                 + 0.936737 w_k10_HSBC_s113
                                 + 0.912382 w_k10_VOD_s113
                                 + 0.87121 w_k10_BP_s113
                                 + 0.887138 w_k10_GSK_s113
                                 + 0.898356 w_k10_AZN_s113
                                 + 0.928921 w_k10_RIO_s113
                                 + 1.14154 w_k10_BG_s113
                                 + 0.941542 w_k10_TSCO_s113
                                 + 0.765113 w_k10_AAL_s113
                                 + 0.715441 w_k10_PRU_s113 >= 1
 z_k10_s1132_constraint:         var_10 + z_k10_s1132
                                 + 0.937365 w_k10_HSBC_s113
                                 + 0.960639 w_k10_VOD_s113
                                 + 0.850628 w_k10_BP_s113
                                 + 0.762355 w_k10_GSK_s113
                                 + 0.97576 w_k10_AZN_s113
                                 + 0.744396 w_k10_RIO_s113
                                 + 0.800554 w_k10_BG_s113
                                 + 0.729198 w_k10_TSCO_s113
                                 + 1.05729 w_k10_AAL_s113
                                 + 0.901984 w_k10_PRU_s113 >= 1
 z_k10_s1133_constraint:         var_10 + z_k10_s1133
                                 + 0.786711 w_k10_HSBC_s113
                                 + 0.980681 w_k10_VOD_s113
                                 + 0.782255 w_k10_BP_s113
                                 + 0.89577 w_k10_GSK_s113
                                 + 1.02897 w_k10_AZN_s113
                                 + 0.80967 w_k10_RIO_s113
                                 + 0.942554 w_k10_BG_s113
                                 + 0.728886 w_k10_TSCO_s113
                                 + 0.939686 w_k10_AAL_s113
                                 + 0.797216 w_k10_PRU_s113 >= 1
 z_k10_s1134_constraint:         var_10 + z_k10_s1134 + 0.81142 w_k10_HSBC_s113
                                 + 0.928435 w_k10_VOD_s113
                                 + 1.02586 w_k10_BP_s113
                                 + 0.886631 w_k10_GSK_s113
                                 + 0.913241 w_k10_AZN_s113
                                 + 0.975952 w_k10_RIO_s113
                                 + 0.874064 w_k10_BG_s113
                                 + 0.885701 w_k10_TSCO_s113
                                 + 1.03872 w_k10_AAL_s113
                                 + 0.947355 w_k10_PRU_s113 >= 1
 z_k10_s1135_constraint:         var_10 + z_k10_s1135 + 1.23718 w_k10_HSBC_s113
                                 + 1.0997 w_k10_VOD_s113 + 1.2775 w_k10_BP_s113
                                 + 1.05742 w_k10_GSK_s113
                                 + 0.982654 w_k10_AZN_s113
                                 + 1.19688 w_k10_RIO_s113
                                 + 1.06493 w_k10_BG_s113
                                 + 1.18427 w_k10_TSCO_s113
                                 + 0.891761 w_k10_AAL_s113
                                 + 1.01076 w_k10_PRU_s113 >= 1
 z_k10_s1136_constraint:         var_10 + z_k10_s1136 + 1.11132 w_k10_HSBC_s113
                                 + 0.920684 w_k10_VOD_s113
                                 + 0.941126 w_k10_BP_s113
                                 + 0.98625 w_k10_GSK_s113
                                 + 0.894857 w_k10_AZN_s113
                                 + 1.01546 w_k10_RIO_s113
                                 + 1.0072 w_k10_BG_s113
                                 + 0.845051 w_k10_TSCO_s113
                                 + 0.921927 w_k10_AAL_s113
                                 + 0.800317 w_k10_PRU_s113 >= 1
 z_k10_s1137_constraint:         var_10 + z_k10_s1137 + 1.13261 w_k10_HSBC_s113
                                 + 1.09073 w_k10_VOD_s113
                                 + 1.23339 w_k10_BP_s113
                                 + 1.02628 w_k10_GSK_s113
                                 + 0.995989 w_k10_AZN_s113
                                 + 1.1206 w_k10_RIO_s113
                                 + 0.973999 w_k10_BG_s113
                                 + 1.13737 w_k10_TSCO_s113
                                 + 1.09557 w_k10_AAL_s113
                                 + 1.27452 w_k10_PRU_s113 >= 1
 z_k10_s1140_constraint:         var_10 + z_k10_s1140
                                 + 0.835432 w_k10_HSBC_s114
                                 + 1.08349 w_k10_VOD_s114
                                 + 0.821978 w_k10_BP_s114
                                 + 0.89643 w_k10_GSK_s114
                                 + 1.10959 w_k10_AZN_s114
                                 + 0.949547 w_k10_RIO_s114
                                 + 1.00848 w_k10_BG_s114
                                 + 0.839845 w_k10_TSCO_s114
                                 + 1.04324 w_k10_AAL_s114
                                 + 1.063 w_k10_PRU_s114 >= 1
 z_k10_s1141_constraint:         var_10 + z_k10_s1141 + 1.24845 w_k10_HSBC_s114
                                 + 1.31152 w_k10_VOD_s114
                                 + 1.24332 w_k10_BP_s114
                                 + 1.06733 w_k10_GSK_s114
                                 + 1.22141 w_k10_AZN_s114
                                 + 1.23055 w_k10_RIO_s114
                                 + 1.03843 w_k10_BG_s114
                                 + 1.17427 w_k10_TSCO_s114
                                 + 1.16 w_k10_AAL_s114 + 1.2437 w_k10_PRU_s114
                                 >= 1
 z_k10_s1142_constraint:         var_10 + z_k10_s1142
                                 + 0.940459 w_k10_HSBC_s114
                                 + 0.989328 w_k10_VOD_s114
                                 + 1.57238 w_k10_BP_s114
                                 + 1.45459 w_k10_GSK_s114
                                 + 1.31096 w_k10_AZN_s114
                                 + 1.28473 w_k10_RIO_s114
                                 + 1.13694 w_k10_BG_s114
                                 + 1.42608 w_k10_TSCO_s114
                                 + 1.073 w_k10_AAL_s114
                                 + 1.10343 w_k10_PRU_s114 >= 1
 z_k10_s1143_constraint:         var_10 + z_k10_s1143 + 1.01516 w_k10_HSBC_s114
                                 + 1.11855 w_k10_VOD_s114
                                 + 1.14319 w_k10_BP_s114
                                 + 1.26463 w_k10_GSK_s114
                                 + 1.3944 w_k10_AZN_s114
                                 + 1.06742 w_k10_RIO_s114
                                 + 1.00846 w_k10_BG_s114
                                 + 1.16435 w_k10_TSCO_s114
                                 + 1.16245 w_k10_AAL_s114
                                 + 1.15491 w_k10_PRU_s114 >= 1
 z_k10_s1144_constraint:         var_10 + z_k10_s1144 + 1.01804 w_k10_HSBC_s114
                                 + 0.944476 w_k10_VOD_s114
                                 + 1.1442 w_k10_BP_s114 + 1.0545 w_k10_GSK_s114
                                 + 1.08673 w_k10_AZN_s114
                                 + 1.04889 w_k10_RIO_s114
                                 + 1.09334 w_k10_BG_s114
                                 + 1.26491 w_k10_TSCO_s114
                                 + 0.980229 w_k10_AAL_s114
                                 + 0.914312 w_k10_PRU_s114 >= 1
 z_k10_s1145_constraint:         var_10 + z_k10_s1145
                                 + 0.978437 w_k10_HSBC_s114
                                 + 0.848346 w_k10_VOD_s114
                                 + 0.789266 w_k10_BP_s114
                                 + 1.00795 w_k10_GSK_s114
                                 + 1.09701 w_k10_AZN_s114
                                 + 0.885576 w_k10_RIO_s114
                                 + 0.878793 w_k10_BG_s114
                                 + 0.824521 w_k10_TSCO_s114
                                 + 1.07549 w_k10_AAL_s114
                                 + 0.88424 w_k10_PRU_s114 >= 1
 z_k10_s1146_constraint:         var_10 + z_k10_s1146
                                 + 0.966615 w_k10_HSBC_s114
                                 + 0.971351 w_k10_VOD_s114
                                 + 0.884369 w_k10_BP_s114
                                 + 0.860116 w_k10_GSK_s114
                                 + 1.07726 w_k10_AZN_s114
                                 + 0.907266 w_k10_RIO_s114
                                 + 1.02359 w_k10_BG_s114
                                 + 0.918585 w_k10_TSCO_s114
                                 + 0.956001 w_k10_AAL_s114
                                 + 0.825971 w_k10_PRU_s114 >= 1
 z_k10_s1147_constraint:         var_10 + z_k10_s1147 + 1.03286 w_k10_HSBC_s114
                                 + 0.962533 w_k10_VOD_s114
                                 + 0.965252 w_k10_BP_s114
                                 + 1.04314 w_k10_GSK_s114
                                 + 0.987429 w_k10_AZN_s114
                                 + 1.01566 w_k10_RIO_s114
                                 + 1.04227 w_k10_BG_s114
                                 + 0.942676 w_k10_TSCO_s114
                                 + 1.11582 w_k10_AAL_s114
                                 + 1.11817 w_k10_PRU_s114 >= 1
 z_k10_s1150_constraint:         var_10 + z_k10_s1150
                                 + 0.757909 w_k10_HSBC_s115
                                 + 0.638436 w_k10_VOD_s115
                                 + 0.750005 w_k10_BP_s115
                                 + 0.862461 w_k10_GSK_s115
                                 + 0.882753 w_k10_AZN_s115
                                 + 0.808095 w_k10_RIO_s115
                                 + 0.994425 w_k10_BG_s115
                                 + 0.928822 w_k10_TSCO_s115
                                 + 1.14415 w_k10_AAL_s115
                                 + 0.948108 w_k10_PRU_s115 >= 1
 z_k10_s1151_constraint:         var_10 + z_k10_s1151 + 1.01762 w_k10_HSBC_s115
                                 + 0.831744 w_k10_VOD_s115
                                 + 1.08672 w_k10_BP_s115
                                 + 1.1723 w_k10_GSK_s115
                                 + 1.03155 w_k10_AZN_s115
                                 + 1.20217 w_k10_RIO_s115
                                 + 1.01282 w_k10_BG_s115
                                 + 1.21592 w_k10_TSCO_s115
                                 + 0.848379 w_k10_AAL_s115
                                 + 1.02955 w_k10_PRU_s115 >= 1
 z_k10_s1152_constraint:         var_10 + z_k10_s1152
                                 + 0.938534 w_k10_HSBC_s115
                                 + 0.909562 w_k10_VOD_s115
                                 + 0.914497 w_k10_BP_s115
                                 + 0.92046 w_k10_GSK_s115
                                 + 0.898474 w_k10_AZN_s115
                                 + 0.990069 w_k10_RIO_s115
                                 + 0.925938 w_k10_BG_s115
                                 + 0.916632 w_k10_TSCO_s115
                                 + 0.918843 w_k10_AAL_s115
                                 + 0.964694 w_k10_PRU_s115 >= 1
 z_k10_s1153_constraint:         var_10 + z_k10_s1153
                                 + 0.829189 w_k10_HSBC_s115
                                 + 0.944226 w_k10_VOD_s115
                                 + 1.08733 w_k10_BP_s115
                                 + 1.05722 w_k10_GSK_s115
                                 + 1.06652 w_k10_AZN_s115
                                 + 1.02472 w_k10_RIO_s115
                                 + 0.918926 w_k10_BG_s115
                                 + 1.03657 w_k10_TSCO_s115
                                 + 1.14094 w_k10_AAL_s115
                                 + 1.28342 w_k10_PRU_s115 >= 1
 z_k10_s1154_constraint:         var_10 + z_k10_s1154 + 1.1306 w_k10_HSBC_s115
                                 + 0.70737 w_k10_VOD_s115
                                 + 1.13366 w_k10_BP_s115
                                 + 1.09766 w_k10_GSK_s115
                                 + 0.918746 w_k10_AZN_s115
                                 + 1.07746 w_k10_RIO_s115
                                 + 1.09145 w_k10_BG_s115
                                 + 1.10341 w_k10_TSCO_s115
                                 + 1.00905 w_k10_AAL_s115
                                 + 0.912099 w_k10_PRU_s115 >= 1
 z_k10_s1155_constraint:         var_10 + z_k10_s1155
                                 + 0.765425 w_k10_HSBC_s115
                                 + 0.679372 w_k10_VOD_s115
                                 + 0.830205 w_k10_BP_s115
                                 + 1.09075 w_k10_GSK_s115
                                 + 0.935675 w_k10_AZN_s115
                                 + 0.931741 w_k10_RIO_s115
                                 + 1.1203 w_k10_BG_s115
                                 + 1.04123 w_k10_TSCO_s115
                                 + 0.820435 w_k10_AAL_s115
                                 + 0.755252 w_k10_PRU_s115 >= 1
 z_k10_s1156_constraint:         var_10 + z_k10_s1156
                                 + 0.874724 w_k10_HSBC_s115
                                 + 0.841251 w_k10_VOD_s115
                                 + 1.1294 w_k10_BP_s115
                                 + 1.02326 w_k10_GSK_s115
                                 + 0.890199 w_k10_AZN_s115
                                 + 1.20138 w_k10_RIO_s115
                                 + 1.33331 w_k10_BG_s115
                                 + 1.26249 w_k10_TSCO_s115
                                 + 0.919063 w_k10_AAL_s115
                                 + 0.862887 w_k10_PRU_s115 >= 1
 z_k10_s1157_constraint:         var_10 + z_k10_s1157 + 1.14161 w_k10_HSBC_s115
                                 + 0.894671 w_k10_VOD_s115
                                 + 1.32262 w_k10_BP_s115
                                 + 1.05382 w_k10_GSK_s115
                                 + 0.93763 w_k10_AZN_s115
                                 + 1.24918 w_k10_RIO_s115
                                 + 0.998828 w_k10_BG_s115
                                 + 1.25795 w_k10_TSCO_s115
                                 + 0.970807 w_k10_AAL_s115
                                 + 1.08415 w_k10_PRU_s115 >= 1
 z_k10_s1160_constraint:         var_10 + z_k10_s1160 + 0.99686 w_k10_HSBC_s116
                                 + 0.969612 w_k10_VOD_s116
                                 + 1.00906 w_k10_BP_s116
                                 + 1.02679 w_k10_GSK_s116
                                 + 1.03967 w_k10_AZN_s116
                                 + 0.959979 w_k10_RIO_s116
                                 + 0.848365 w_k10_BG_s116
                                 + 0.917888 w_k10_TSCO_s116
                                 + 1.20222 w_k10_AAL_s116
                                 + 1.23016 w_k10_PRU_s116 >= 1
 z_k10_s1161_constraint:         var_10 + z_k10_s1161 + 1.02132 w_k10_HSBC_s116
                                 + 1.09835 w_k10_VOD_s116
                                 + 0.9249 w_k10_BP_s116
                                 + 0.892702 w_k10_GSK_s116
                                 + 0.962102 w_k10_AZN_s116
                                 + 0.939516 w_k10_RIO_s116
                                 + 0.957704 w_k10_BG_s116
                                 + 0.799881 w_k10_TSCO_s116
                                 + 0.866787 w_k10_AAL_s116
                                 + 0.915742 w_k10_PRU_s116 >= 1
 z_k10_s1162_constraint:         var_10 + z_k10_s1162 + 1.13367 w_k10_HSBC_s116
                                 + 0.913581 w_k10_VOD_s116
                                 + 0.742225 w_k10_BP_s116
                                 + 0.789446 w_k10_GSK_s116
                                 + 0.963589 w_k10_AZN_s116
                                 + 0.750517 w_k10_RIO_s116
                                 + 0.828152 w_k10_BG_s116
                                 + 0.71075 w_k10_TSCO_s116
                                 + 0.893992 w_k10_AAL_s116
                                 + 0.985056 w_k10_PRU_s116 >= 1
 z_k10_s1163_constraint:         var_10 + z_k10_s1163 + 1.23041 w_k10_HSBC_s116
                                 + 0.996225 w_k10_VOD_s116
                                 + 1.38588 w_k10_BP_s116
                                 + 1.08632 w_k10_GSK_s116
                                 + 0.883239 w_k10_AZN_s116
                                 + 1.21993 w_k10_RIO_s116
                                 + 1.18914 w_k10_BG_s116
                                 + 1.26469 w_k10_TSCO_s116
                                 + 0.902631 w_k10_AAL_s116
                                 + 0.951299 w_k10_PRU_s116 >= 1
 z_k10_s1164_constraint:         var_10 + z_k10_s1164 + 1.06796 w_k10_HSBC_s116
                                 + 1.05368 w_k10_VOD_s116
                                 + 1.15182 w_k10_BP_s116
                                 + 1.03659 w_k10_GSK_s116
                                 + 0.938288 w_k10_AZN_s116
                                 + 1.03499 w_k10_RIO_s116
                                 + 0.946022 w_k10_BG_s116
                                 + 1.15912 w_k10_TSCO_s116
                                 + 1.02002 w_k10_AAL_s116
                                 + 1.02935 w_k10_PRU_s116 >= 1
 z_k10_s1165_constraint:         var_10 + z_k10_s1165 + 1.29181 w_k10_HSBC_s116
                                 + 1.10819 w_k10_VOD_s116
                                 + 1.02238 w_k10_BP_s116
                                 + 1.02982 w_k10_GSK_s116
                                 + 0.997591 w_k10_AZN_s116
                                 + 1.02616 w_k10_RIO_s116
                                 + 1.02457 w_k10_BG_s116
                                 + 0.916315 w_k10_TSCO_s116
                                 + 0.947624 w_k10_AAL_s116
                                 + 1.03558 w_k10_PRU_s116 >= 1
 z_k10_s1166_constraint:         var_10 + z_k10_s1166 + 0.97652 w_k10_HSBC_s116
                                 + 0.802802 w_k10_VOD_s116
                                 + 0.821572 w_k10_BP_s116
                                 + 1.02567 w_k10_GSK_s116
                                 + 0.919579 w_k10_AZN_s116
                                 + 0.924241 w_k10_RIO_s116
                                 + 1.11133 w_k10_BG_s116
                                 + 0.902546 w_k10_TSCO_s116
                                 + 0.854723 w_k10_AAL_s116
                                 + 0.760418 w_k10_PRU_s116 >= 1
 z_k10_s1167_constraint:         var_10 + z_k10_s1167
                                 + 0.768203 w_k10_HSBC_s116
                                 + 0.831932 w_k10_VOD_s116
                                 + 0.787613 w_k10_BP_s116
                                 + 0.858116 w_k10_GSK_s116
                                 + 0.961231 w_k10_AZN_s116
                                 + 0.79278 w_k10_RIO_s116
                                 + 0.894153 w_k10_BG_s116
                                 + 0.806902 w_k10_TSCO_s116
                                 + 1.03667 w_k10_AAL_s116
                                 + 0.981332 w_k10_PRU_s116 >= 1
 z_k10_s1170_constraint:         var_10 + z_k10_s1170 + 0.76317 w_k10_HSBC_s117
                                 + 0.780091 w_k10_VOD_s117
                                 + 0.865767 w_k10_BP_s117
                                 + 0.975727 w_k10_GSK_s117
                                 + 0.79162 w_k10_AZN_s117
                                 + 0.911903 w_k10_RIO_s117
                                 + 0.978226 w_k10_BG_s117
                                 + 0.910737 w_k10_TSCO_s117
                                 + 0.997265 w_k10_AAL_s117
                                 + 0.998257 w_k10_PRU_s117 >= 1
 z_k10_s1171_constraint:         var_10 + z_k10_s1171
                                 + 0.879276 w_k10_HSBC_s117
                                 + 0.968309 w_k10_VOD_s117
                                 + 0.893134 w_k10_BP_s117
                                 + 0.851429 w_k10_GSK_s117
                                 + 0.953979 w_k10_AZN_s117
                                 + 0.825015 w_k10_RIO_s117
                                 + 0.863504 w_k10_BG_s117
                                 + 0.792978 w_k10_TSCO_s117
                                 + 1.13077 w_k10_AAL_s117
                                 + 1.0054 w_k10_PRU_s117 >= 1
 z_k10_s1172_constraint:         var_10 + z_k10_s1172 + 1.01355 w_k10_HSBC_s117
                                 + 1.00251 w_k10_VOD_s117
                                 + 1.12058 w_k10_BP_s117
                                 + 1.05167 w_k10_GSK_s117
                                 + 0.950368 w_k10_AZN_s117
                                 + 0.9752 w_k10_RIO_s117
                                 + 0.965788 w_k10_BG_s117
                                 + 0.974195 w_k10_TSCO_s117
                                 + 1.12602 w_k10_AAL_s117
                                 + 1.04405 w_k10_PRU_s117 >= 1
 z_k10_s1173_constraint:         var_10 + z_k10_s1173 + 1.00973 w_k10_HSBC_s117
                                 + 1.00163 w_k10_VOD_s117
                                 + 1.01326 w_k10_BP_s117
                                 + 1.03635 w_k10_GSK_s117
                                 + 0.969059 w_k10_AZN_s117
                                 + 0.978183 w_k10_RIO_s117
                                 + 0.860339 w_k10_BG_s117
                                 + 1.02245 w_k10_TSCO_s117
                                 + 1.02931 w_k10_AAL_s117
                                 + 1.15878 w_k10_PRU_s117 >= 1
 z_k10_s1174_constraint:         var_10 + z_k10_s1174 + 1.15012 w_k10_HSBC_s117
                                 + 1.06919 w_k10_VOD_s117
                                 + 1.41377 w_k10_BP_s117
                                 + 1.33265 w_k10_GSK_s117
                                 + 1.00949 w_k10_AZN_s117
                                 + 1.24113 w_k10_RIO_s117
                                 + 1.07478 w_k10_BG_s117
                                 + 1.3697 w_k10_TSCO_s117
                                 + 1.08536 w_k10_AAL_s117
                                 + 1.1281 w_k10_PRU_s117 >= 1
 z_k10_s1175_constraint:         var_10 + z_k10_s1175 + 0.89687 w_k10_HSBC_s117
                                 + 0.699512 w_k10_VOD_s117
                                 + 0.884104 w_k10_BP_s117
                                 + 0.876607 w_k10_GSK_s117
                                 + 0.781439 w_k10_AZN_s117
                                 + 0.826673 w_k10_RIO_s117
                                 + 1.15279 w_k10_BG_s117
                                 + 0.882497 w_k10_TSCO_s117
                                 + 0.987066 w_k10_AAL_s117
                                 + 0.845567 w_k10_PRU_s117 >= 1
 z_k10_s1176_constraint:         var_10 + z_k10_s1176 + 1.02494 w_k10_HSBC_s117
                                 + 0.92221 w_k10_VOD_s117
                                 + 1.07505 w_k10_BP_s117
                                 + 1.05779 w_k10_GSK_s117
                                 + 0.896321 w_k10_AZN_s117
                                 + 1.10064 w_k10_RIO_s117
                                 + 1.25351 w_k10_BG_s117
                                 + 1.06314 w_k10_TSCO_s117
                                 + 0.920304 w_k10_AAL_s117
                                 + 0.893248 w_k10_PRU_s117 >= 1
 z_k10_s1177_constraint:         var_10 + z_k10_s1177 + 1.1936 w_k10_HSBC_s117
                                 + 0.866426 w_k10_VOD_s117
                                 + 0.991647 w_k10_BP_s117
                                 + 1.04689 w_k10_GSK_s117
                                 + 0.883395 w_k10_AZN_s117
                                 + 1.16058 w_k10_RIO_s117
                                 + 0.991825 w_k10_BG_s117
                                 + 1.09391 w_k10_TSCO_s117
                                 + 0.840672 w_k10_AAL_s117
                                 + 0.970037 w_k10_PRU_s117 >= 1
 k10_return_constraint:          0.0115519467560434 w_k10_HSBC_s000
                                 + 0.0161477844805614 w_k10_HSBC_s001
                                 + 0.0169958690446538 w_k10_HSBC_s002
                                 + 0.0270368179127118 w_k10_HSBC_s003
                                 + 0.0257195622359307 w_k10_HSBC_s004
                                 + 0.0142427692006247 w_k10_HSBC_s005
                                 + 0.016634862049119 w_k10_HSBC_s006
                                 + 0.0195622480755351 w_k10_HSBC_s007
                                 + 0.0211861911640295 w_k10_HSBC_s010
                                 + 0.0163690186169615 w_k10_HSBC_s011
                                 + 0.065090272495904 w_k10_HSBC_s012
                                 + 0.0176045738826572 w_k10_HSBC_s013
                                 + 0.0621274887743174 w_k10_HSBC_s014
                                 + 0.0174049218092407 w_k10_HSBC_s015
                                 + 0.050400505612286 w_k10_HSBC_s016
                                 + 0.0174244959199363 w_k10_HSBC_s017
                                 + 0.0306115168863649 w_k10_HSBC_s100
                                 + 0.0579746123267569 w_k10_HSBC_s101
                                 + 0.093948154523141 w_k10_HSBC_s102
                                 + 0.0772016973528142 w_k10_HSBC_s103
                                 + 0.0891618001700647 w_k10_HSBC_s104
                                 + 0.0545809569116311 w_k10_HSBC_s105
                                 + 0.0227925925513981 w_k10_HSBC_s106
                                 + 0.036520620404593 w_k10_HSBC_s107
                                 + 0.011353216240556 w_k10_HSBC_s110
                                 + 0.0276909154963832 w_k10_HSBC_s111
                                 + 0.00798564572718046 w_k10_HSBC_s112
                                 + 0.0125170453614022 w_k10_HSBC_s113
                                 + 0.0111143580369364 w_k10_HSBC_s114
                                 + 0.0121064634830739 w_k10_HSBC_s115
                                 + 0.0275007698746671 w_k10_HSBC_s116
                                 + 0.0235130684452349 w_k10_HSBC_s117
                                 + 0.0147694494812484 w_k10_VOD_s000
                                 + 0.0181558331067359 w_k10_VOD_s001
                                 + 0.0176647007613049 w_k10_VOD_s002
                                 + 0.0260223784798861 w_k10_VOD_s003
                                 + 0.0266578037591939 w_k10_VOD_s004
                                 + 0.0130251069337626 w_k10_VOD_s005
                                 + 0.0161495650328867 w_k10_VOD_s006
                                 + 0.0186209359662451 w_k10_VOD_s007
                                 + 0.0208029128343992 w_k10_VOD_s010
                                 + 0.0172769192570446 w_k10_VOD_s011
                                 + 0.0632421701754749 w_k10_VOD_s012
                                 + 0.0154301860730499 w_k10_VOD_s013
                                 + 0.0598029509800832 w_k10_VOD_s014
                                 + 0.0182336194087219 w_k10_VOD_s015
                                 + 0.0544482419909423 w_k10_VOD_s016
                                 + 0.0148858854701797 w_k10_VOD_s017
                                 + 0.0331640014939138 w_k10_VOD_s100
                                 + 0.0606743786484985 w_k10_VOD_s101
                                 + 0.0952350306903382 w_k10_VOD_s102
                                 + 0.0817822708745166 w_k10_VOD_s103
                                 + 0.085228519270808 w_k10_VOD_s104
                                 + 0.0545958592881511 w_k10_VOD_s105
                                 + 0.025233645951723 w_k10_VOD_s106
                                 + 0.036031983236384 w_k10_VOD_s107
                                 + 0.0102452848926923 w_k10_VOD_s110
                                 + 0.024691005125294 w_k10_VOD_s111
                                 + 0.00692869990798533 w_k10_VOD_s112
                                 + 0.012246984381081 w_k10_VOD_s113
                                 + 0.0112413392012356 w_k10_VOD_s114
                                 + 0.0107458953435676 w_k10_VOD_s115
                                 + 0.025192102579319 w_k10_VOD_s116
                                 + 0.0219202150675142 w_k10_VOD_s117
                                 + 0.0117490844304173 w_k10_BP_s000
                                 + 0.0166723320135281 w_k10_BP_s001
                                 + 0.0167719789068041 w_k10_BP_s002
                                 + 0.0270122133180271 w_k10_BP_s003
                                 + 0.0258624121522477 w_k10_BP_s004
                                 + 0.0142960518454211 w_k10_BP_s005
                                 + 0.0179300652192248 w_k10_BP_s006
                                 + 0.0187838627483177 w_k10_BP_s007
                                 + 0.0195850285752395 w_k10_BP_s010
                                 + 0.0170082218815417 w_k10_BP_s011
                                 + 0.0635105074636541 w_k10_BP_s012
                                 + 0.0172003161928268 w_k10_BP_s013
                                 + 0.0602030985247861 w_k10_BP_s014
                                 + 0.0179377099939513 w_k10_BP_s015
                                 + 0.0564912883870212 w_k10_BP_s016
                                 + 0.0173186657822625 w_k10_BP_s017
                                 + 0.0320396601842605 w_k10_BP_s100
                                 + 0.0620379479846927 w_k10_BP_s101
                                 + 0.0905120259703476 w_k10_BP_s102
                                 + 0.0790386228479708 w_k10_BP_s103
                                 + 0.0860650915364934 w_k10_BP_s104
                                 + 0.060015225451287 w_k10_BP_s105
                                 + 0.0257263476271255 w_k10_BP_s106
                                 + 0.0358927073043478 w_k10_BP_s107
                                 + 0.0110878331996637 w_k10_BP_s110
                                 + 0.0295075865119546 w_k10_BP_s111
                                 + 0.00732742967504155 w_k10_BP_s112
                                 + 0.0128724194825491 w_k10_BP_s113
                                 + 0.0115520554333819 w_k10_BP_s114
                                 + 0.0135715109785616 w_k10_BP_s115
                                 + 0.0265204709411625 w_k10_BP_s116
                                 + 0.0249255017503052 w_k10_BP_s117
                                 + 0.0114659257033723 w_k10_GSK_s000
                                 + 0.016378880843971 w_k10_GSK_s001
                                 + 0.0161947636044197 w_k10_GSK_s002
                                 + 0.0270814240401156 w_k10_GSK_s003
                                 + 0.0217606546479536 w_k10_GSK_s004
                                 + 0.014336724527016 w_k10_GSK_s005
                                 + 0.0157429824312703 w_k10_GSK_s006
                                 + 0.0200151050217179 w_k10_GSK_s007
                                 + 0.020864421866814 w_k10_GSK_s010
                                 + 0.0171076888009321 w_k10_GSK_s011
                                 + 0.0653236044611027 w_k10_GSK_s012
                                 + 0.0159989129748705 w_k10_GSK_s013
                                 + 0.0594928949535413 w_k10_GSK_s014
                                 + 0.0201656634108191 w_k10_GSK_s015
                                 + 0.0538415980366381 w_k10_GSK_s016
                                 + 0.0157965741391778 w_k10_GSK_s017
                                 + 0.0307354979796209 w_k10_GSK_s100
                                 + 0.0553549690118524 w_k10_GSK_s101
                                 + 0.0888137184761519 w_k10_GSK_s102
                                 + 0.0816511657503805 w_k10_GSK_s103
                                 + 0.0825690212325732 w_k10_GSK_s104
                                 + 0.0597116181531604 w_k10_GSK_s105
                                 + 0.0249834557501263 w_k10_GSK_s106
                                 + 0.0364260902360001 w_k10_GSK_s107
                                 + 0.0103923821548677 w_k10_GSK_s110
                                 + 0.0271279661640002 w_k10_GSK_s111
                                 + 0.00725127920606892 w_k10_GSK_s112
                                 + 0.0116231241282724 w_k10_GSK_s113
                                 + 0.0117633567829389 w_k10_GSK_s114
                                 + 0.0133786893885875 w_k10_GSK_s115
                                 + 0.0257670895012214 w_k10_GSK_s116
                                 + 0.024845535903465 w_k10_GSK_s117
                                 + 0.0128371079170183 w_k10_AZN_s000
                                 + 0.0172490794226395 w_k10_AZN_s001
                                 + 0.0157061498190709 w_k10_AZN_s002
                                 + 0.0299983870502552 w_k10_AZN_s003
                                 + 0.024170684166881 w_k10_AZN_s004
                                 + 0.0131952588338207 w_k10_AZN_s005
                                 + 0.0162192278280214 w_k10_AZN_s006
                                 + 0.0193600542808604 w_k10_AZN_s007
                                 + 0.0210133412686277 w_k10_AZN_s010
                                 + 0.0177133682493139 w_k10_AZN_s011
                                 + 0.0644651393825505 w_k10_AZN_s012
                                 + 0.0168222456676095 w_k10_AZN_s013
                                 + 0.0573915653694943 w_k10_AZN_s014
                                 + 0.0193466560500078 w_k10_AZN_s015
                                 + 0.0478335610716706 w_k10_AZN_s016
                                 + 0.0163093126955718 w_k10_AZN_s017
                                 + 0.0330248896995505 w_k10_AZN_s100
                                 + 0.0583007610008679 w_k10_AZN_s101
                                 + 0.0932303256051328 w_k10_AZN_s102
                                 + 0.0788181589357633 w_k10_AZN_s103
                                 + 0.0836070015565014 w_k10_AZN_s104
                                 + 0.0605420955657598 w_k10_AZN_s105
                                 + 0.0239750293902721 w_k10_AZN_s106
                                 + 0.0374134882451913 w_k10_AZN_s107
                                 + 0.00955261711304138 w_k10_AZN_s110
                                 + 0.0249352193237179 w_k10_AZN_s111
                                 + 0.00708636124414273 w_k10_AZN_s112
                                 + 0.0115753634786186 w_k10_AZN_s113
                                 + 0.0126505988436143 w_k10_AZN_s114
                                 + 0.0123604193947819 w_k10_AZN_s115
                                 + 0.024799067111242 w_k10_AZN_s116
                                 + 0.0218261899829328 w_k10_AZN_s117
                                 + 0.0128269231701131 w_k10_RIO_s000
                                 + 0.0173783456127057 w_k10_RIO_s001
                                 + 0.0170684444989011 w_k10_RIO_s002
                                 + 0.0248234813105686 w_k10_RIO_s003
                                 + 0.0255369127824715 w_k10_RIO_s004
                                 + 0.0137348241529025 w_k10_RIO_s005
                                 + 0.0171369180238897 w_k10_RIO_s006
                                 + 0.018607004079876 w_k10_RIO_s007
                                 + 0.0199420208183499 w_k10_RIO_s010
                                 + 0.0181399282551862 w_k10_RIO_s011
                                 + 0.0617630011240528 w_k10_RIO_s012
                                 + 0.0151952467513449 w_k10_RIO_s013
                                 + 0.057738463747233 w_k10_RIO_s014
                                 + 0.0178752439620105 w_k10_RIO_s015
                                 + 0.0569770741840445 w_k10_RIO_s016
                                 + 0.0168769107264823 w_k10_RIO_s017
                                 + 0.0308540148099308 w_k10_RIO_s100
                                 + 0.0629738396458573 w_k10_RIO_s101
                                 + 0.0955696932771688 w_k10_RIO_s102
                                 + 0.0792285272405526 w_k10_RIO_s103
                                 + 0.0867099878081349 w_k10_RIO_s104
                                 + 0.0592753710064017 w_k10_RIO_s105
                                 + 0.0239019296647679 w_k10_RIO_s106
                                 + 0.0362686938617191 w_k10_RIO_s107
                                 + 0.010385995898313 w_k10_RIO_s110
                                 + 0.0272223688066016 w_k10_RIO_s111
                                 + 0.00789241000580985 w_k10_RIO_s112
                                 + 0.012384303723136 w_k10_RIO_s113
                                 + 0.0114258402118475 w_k10_RIO_s114
                                 + 0.0137537932757108 w_k10_RIO_s115
                                 + 0.0254856447431202 w_k10_RIO_s116
                                 + 0.0239488740307602 w_k10_RIO_s117
                                 + 0.0120778074766737 w_k10_BG_s000
                                 + 0.0163750946249541 w_k10_BG_s001
                                 + 0.0169482230150286 w_k10_BG_s002
                                 + 0.0261470249721652 w_k10_BG_s003
                                 + 0.0246714493056887 w_k10_BG_s004
                                 + 0.0149138547557082 w_k10_BG_s005
                                 + 0.0167267757896612 w_k10_BG_s006
                                 + 0.0186171189338762 w_k10_BG_s007
                                 + 0.019603130142445 w_k10_BG_s010
                                 + 0.0169708888937868 w_k10_BG_s011
                                 + 0.0620828170231441 w_k10_BG_s012
                                 + 0.0168891464488967 w_k10_BG_s013
                                 + 0.0614893141425234 w_k10_BG_s014
                                 + 0.0176981576493793 w_k10_BG_s015
                                 + 0.0531931315238203 w_k10_BG_s016
                                 + 0.0166587290852196 w_k10_BG_s017
                                 + 0.0307724186759419 w_k10_BG_s100
                                 + 0.059519763290701 w_k10_BG_s101
                                 + 0.0891148748267454 w_k10_BG_s102
                                 + 0.0772031212598205 w_k10_BG_s103
                                 + 0.0836578124249484 w_k10_BG_s104
                                 + 0.058526642709318 w_k10_BG_s105
                                 + 0.023589849497391 w_k10_BG_s106
                                 + 0.0380791662627279 w_k10_BG_s107
                                 + 0.010752587828786 w_k10_BG_s110
                                 + 0.0271188666494118 w_k10_BG_s111
                                 + 0.00751514915525919 w_k10_BG_s112
                                 + 0.0120920652745402 w_k10_BG_s113
                                 + 0.0113712580134958 w_k10_BG_s114
                                 + 0.013200576680273 w_k10_BG_s115
                                 + 0.0256028803128873 w_k10_BG_s116
                                 + 0.0255103787652978 w_k10_BG_s117
                                 + 0.0118899150035675 w_k10_TSCO_s000
                                 + 0.0167866835794908 w_k10_TSCO_s001
                                 + 0.0163159015565307 w_k10_TSCO_s002
                                 + 0.0283407565646272 w_k10_TSCO_s003
                                 + 0.0246734205320971 w_k10_TSCO_s004
                                 + 0.016626825476399 w_k10_TSCO_s005
                                 + 0.018053532488431 w_k10_TSCO_s006
                                 + 0.0187288299649796 w_k10_TSCO_s007
                                 + 0.0219975975804409 w_k10_TSCO_s010
                                 + 0.0168057900353342 w_k10_TSCO_s011
                                 + 0.067226730028072 w_k10_TSCO_s012
                                 + 0.0175606703590326 w_k10_TSCO_s013
                                 + 0.057635541393835 w_k10_TSCO_s014
                                 + 0.0200352853679499 w_k10_TSCO_s015
                                 + 0.0535631533103183 w_k10_TSCO_s016
                                 + 0.0180940179762416 w_k10_TSCO_s017
                                 + 0.0335615043686089 w_k10_TSCO_s100
                                 + 0.0642577503616773 w_k10_TSCO_s101
                                 + 0.0911606523324736 w_k10_TSCO_s102
                                 + 0.0736990385221625 w_k10_TSCO_s103
                                 + 0.0907634775174408 w_k10_TSCO_s104
                                 + 0.0595747535821971 w_k10_TSCO_s105
                                 + 0.0259473585677479 w_k10_TSCO_s106
                                 + 0.0358774452863498 w_k10_TSCO_s107
                                 + 0.0110966927154991 w_k10_TSCO_s110
                                 + 0.0297195062122497 w_k10_TSCO_s111
                                 + 0.00750211359135583 w_k10_TSCO_s112
                                 + 0.0118978323222656 w_k10_TSCO_s113
                                 + 0.0117257308330983 w_k10_TSCO_s114
                                 + 0.0140475616090555 w_k10_TSCO_s115
                                 + 0.0255014947332771 w_k10_TSCO_s116
                                 + 0.0243277406749103 w_k10_TSCO_s117
                                 + 0.0122391520458497 w_k10_AAL_s000
                                 + 0.0164393018226452 w_k10_AAL_s001
                                 + 0.0166105763238446 w_k10_AAL_s002
                                 + 0.0283565943023726 w_k10_AAL_s003
                                 + 0.024538460849685 w_k10_AAL_s004
                                 + 0.01419489984014 w_k10_AAL_s005
                                 + 0.0169443824757084 w_k10_AAL_s006
                                 + 0.0198367881094659 w_k10_AAL_s007
                                 + 0.0205513130161859 w_k10_AAL_s010
                                 + 0.0182239984743004 w_k10_AAL_s011
                                 + 0.0634168267903503 w_k10_AAL_s012
                                 + 0.0172152438170201 w_k10_AAL_s013
                                 + 0.0630700750025073 w_k10_AAL_s014
                                 + 0.0188132015409845 w_k10_AAL_s015
                                 + 0.0548546917191425 w_k10_AAL_s016
                                 + 0.0166948510498845 w_k10_AAL_s017
                                 + 0.0329531050555003 w_k10_AAL_s100
                                 + 0.061478359004499 w_k10_AAL_s101
                                 + 0.0866876618856424 w_k10_AAL_s102
                                 + 0.0763929811647 w_k10_AAL_s103
                                 + 0.0817717564404668 w_k10_AAL_s104
                                 + 0.0582377752092093 w_k10_AAL_s105
                                 + 0.023935695633945 w_k10_AAL_s106
                                 + 0.0359794222814958 w_k10_AAL_s107
                                 + 0.0102821249947416 w_k10_AAL_s110
                                 + 0.02789316944591 w_k10_AAL_s111
                                 + 0.0075356782704078 w_k10_AAL_s112
                                 + 0.0114123294381223 w_k10_AAL_s113
                                 + 0.0117947063939554 w_k10_AAL_s114
                                 + 0.0127125671233161 w_k10_AAL_s115
                                 + 0.0254803573736185 w_k10_AAL_s116
                                 + 0.0250581063721294 w_k10_AAL_s117
                                 + 0.0131079246621183 w_k10_PRU_s000
                                 + 0.0187090054525294 w_k10_PRU_s001
                                 + 0.0138684812383071 w_k10_PRU_s002
                                 + 0.030540848816043 w_k10_PRU_s003
                                 + 0.0271528898304193 w_k10_PRU_s004
                                 + 0.0135422908541362 w_k10_PRU_s005
                                 + 0.0186097612999828 w_k10_PRU_s006
                                 + 0.0190077573498461 w_k10_PRU_s007
                                 + 0.0203103296259024 w_k10_PRU_s010
                                 + 0.018352420995655 w_k10_PRU_s011
                                 + 0.0629582428397341 w_k10_PRU_s012
                                 + 0.0154802735641576 w_k10_PRU_s013
                                 + 0.0669505282337757 w_k10_PRU_s014
                                 + 0.0179536546250521 w_k10_PRU_s015
                                 + 0.0533746780686088 w_k10_PRU_s016
                                 + 0.0164052824500929 w_k10_PRU_s017
                                 + 0.0310489267222267 w_k10_PRU_s100
                                 + 0.0709792637646171 w_k10_PRU_s101
                                 + 0.0946991499225905 w_k10_PRU_s102
                                 + 0.0716678508031369 w_k10_PRU_s103
                                 + 0.0734986674258189 w_k10_PRU_s104
                                 + 0.0649123199309517 w_k10_PRU_s105
                                 + 0.0231666357039322 w_k10_PRU_s106
                                 + 0.0393321802284137 w_k10_PRU_s107
                                 + 0.010715394267472 w_k10_PRU_s110
                                 + 0.0266944794288319 w_k10_PRU_s111
                                 + 0.00798965029784652 w_k10_PRU_s112
                                 + 0.0115434878066284 w_k10_PRU_s113
                                 + 0.0113525986895928 w_k10_PRU_s114
                                 + 0.0131658815746629 w_k10_PRU_s115
                                 + 0.0258159746741073 w_k10_PRU_s116
                                 + 0.0242453105906263 w_k10_PRU_s117 >= 1.1
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
 0 <= w_k10_HSBC_s00 <= 1
 0 <= w_k10_HSBC_s01 <= 1
 0 <= w_k10_HSBC_s10 <= 1
 0 <= w_k10_HSBC_s11 <= 1
 0 <= w_k10_HSBC_s000 <= 1
 0 <= w_k10_HSBC_s001 <= 1
 0 <= w_k10_HSBC_s002 <= 1
 0 <= w_k10_HSBC_s003 <= 1
 0 <= w_k10_HSBC_s004 <= 1
 0 <= w_k10_HSBC_s005 <= 1
 0 <= w_k10_HSBC_s006 <= 1
 0 <= w_k10_HSBC_s007 <= 1
 0 <= w_k10_HSBC_s010 <= 1
 0 <= w_k10_HSBC_s011 <= 1
 0 <= w_k10_HSBC_s012 <= 1
 0 <= w_k10_HSBC_s013 <= 1
 0 <= w_k10_HSBC_s014 <= 1
 0 <= w_k10_HSBC_s015 <= 1
 0 <= w_k10_HSBC_s016 <= 1
 0 <= w_k10_HSBC_s017 <= 1
 0 <= w_k10_HSBC_s100 <= 1
 0 <= w_k10_HSBC_s101 <= 1
 0 <= w_k10_HSBC_s102 <= 1
 0 <= w_k10_HSBC_s103 <= 1
 0 <= w_k10_HSBC_s104 <= 1
 0 <= w_k10_HSBC_s105 <= 1
 0 <= w_k10_HSBC_s106 <= 1
 0 <= w_k10_HSBC_s107 <= 1
 0 <= w_k10_HSBC_s110 <= 1
 0 <= w_k10_HSBC_s111 <= 1
 0 <= w_k10_HSBC_s112 <= 1
 0 <= w_k10_HSBC_s113 <= 1
 0 <= w_k10_HSBC_s114 <= 1
 0 <= w_k10_HSBC_s115 <= 1
 0 <= w_k10_HSBC_s116 <= 1
 0 <= w_k10_HSBC_s117 <= 1
 0 <= w_k10_VOD_s0 <= 1
 0 <= w_k10_VOD_s1 <= 1
 0 <= w_k10_VOD_s00 <= 1
 0 <= w_k10_VOD_s01 <= 1
 0 <= w_k10_VOD_s10 <= 1
 0 <= w_k10_VOD_s11 <= 1
 0 <= w_k10_VOD_s000 <= 1
 0 <= w_k10_VOD_s001 <= 1
 0 <= w_k10_VOD_s002 <= 1
 0 <= w_k10_VOD_s003 <= 1
 0 <= w_k10_VOD_s004 <= 1
 0 <= w_k10_VOD_s005 <= 1
 0 <= w_k10_VOD_s006 <= 1
 0 <= w_k10_VOD_s007 <= 1
 0 <= w_k10_VOD_s010 <= 1
 0 <= w_k10_VOD_s011 <= 1
 0 <= w_k10_VOD_s012 <= 1
 0 <= w_k10_VOD_s013 <= 1
 0 <= w_k10_VOD_s014 <= 1
 0 <= w_k10_VOD_s015 <= 1
 0 <= w_k10_VOD_s016 <= 1
 0 <= w_k10_VOD_s017 <= 1
 0 <= w_k10_VOD_s100 <= 1
 0 <= w_k10_VOD_s101 <= 1
 0 <= w_k10_VOD_s102 <= 1
 0 <= w_k10_VOD_s103 <= 1
 0 <= w_k10_VOD_s104 <= 1
 0 <= w_k10_VOD_s105 <= 1
 0 <= w_k10_VOD_s106 <= 1
 0 <= w_k10_VOD_s107 <= 1
 0 <= w_k10_VOD_s110 <= 1
 0 <= w_k10_VOD_s111 <= 1
 0 <= w_k10_VOD_s112 <= 1
 0 <= w_k10_VOD_s113 <= 1
 0 <= w_k10_VOD_s114 <= 1
 0 <= w_k10_VOD_s115 <= 1
 0 <= w_k10_VOD_s116 <= 1
 0 <= w_k10_VOD_s117 <= 1
 0 <= w_k10_BP_s0 <= 1
 0 <= w_k10_BP_s1 <= 1
 0 <= w_k10_BP_s00 <= 1
 0 <= w_k10_BP_s01 <= 1
 0 <= w_k10_BP_s10 <= 1
 0 <= w_k10_BP_s11 <= 1
 0 <= w_k10_BP_s000 <= 1
 0 <= w_k10_BP_s001 <= 1
 0 <= w_k10_BP_s002 <= 1
 0 <= w_k10_BP_s003 <= 1
 0 <= w_k10_BP_s004 <= 1
 0 <= w_k10_BP_s005 <= 1
 0 <= w_k10_BP_s006 <= 1
 0 <= w_k10_BP_s007 <= 1
 0 <= w_k10_BP_s010 <= 1
 0 <= w_k10_BP_s011 <= 1
 0 <= w_k10_BP_s012 <= 1
 0 <= w_k10_BP_s013 <= 1
 0 <= w_k10_BP_s014 <= 1
 0 <= w_k10_BP_s015 <= 1
 0 <= w_k10_BP_s016 <= 1
 0 <= w_k10_BP_s017 <= 1
 0 <= w_k10_BP_s100 <= 1
 0 <= w_k10_BP_s101 <= 1
 0 <= w_k10_BP_s102 <= 1
 0 <= w_k10_BP_s103 <= 1
 0 <= w_k10_BP_s104 <= 1
 0 <= w_k10_BP_s105 <= 1
 0 <= w_k10_BP_s106 <= 1
 0 <= w_k10_BP_s107 <= 1
 0 <= w_k10_BP_s110 <= 1
 0 <= w_k10_BP_s111 <= 1
 0 <= w_k10_BP_s112 <= 1
 0 <= w_k10_BP_s113 <= 1
 0 <= w_k10_BP_s114 <= 1
 0 <= w_k10_BP_s115 <= 1
 0 <= w_k10_BP_s116 <= 1
 0 <= w_k10_BP_s117 <= 1
 0 <= w_k10_GSK_s0 <= 1
 0 <= w_k10_GSK_s1 <= 1
 0 <= w_k10_GSK_s00 <= 1
 0 <= w_k10_GSK_s01 <= 1
 0 <= w_k10_GSK_s10 <= 1
 0 <= w_k10_GSK_s11 <= 1
 0 <= w_k10_GSK_s000 <= 1
 0 <= w_k10_GSK_s001 <= 1
 0 <= w_k10_GSK_s002 <= 1
 0 <= w_k10_GSK_s003 <= 1
 0 <= w_k10_GSK_s004 <= 1
 0 <= w_k10_GSK_s005 <= 1
 0 <= w_k10_GSK_s006 <= 1
 0 <= w_k10_GSK_s007 <= 1
 0 <= w_k10_GSK_s010 <= 1
 0 <= w_k10_GSK_s011 <= 1
 0 <= w_k10_GSK_s012 <= 1
 0 <= w_k10_GSK_s013 <= 1
 0 <= w_k10_GSK_s014 <= 1
 0 <= w_k10_GSK_s015 <= 1
 0 <= w_k10_GSK_s016 <= 1
 0 <= w_k10_GSK_s017 <= 1
 0 <= w_k10_GSK_s100 <= 1
 0 <= w_k10_GSK_s101 <= 1
 0 <= w_k10_GSK_s102 <= 1
 0 <= w_k10_GSK_s103 <= 1
 0 <= w_k10_GSK_s104 <= 1
 0 <= w_k10_GSK_s105 <= 1
 0 <= w_k10_GSK_s106 <= 1
 0 <= w_k10_GSK_s107 <= 1
 0 <= w_k10_GSK_s110 <= 1
 0 <= w_k10_GSK_s111 <= 1
 0 <= w_k10_GSK_s112 <= 1
 0 <= w_k10_GSK_s113 <= 1
 0 <= w_k10_GSK_s114 <= 1
 0 <= w_k10_GSK_s115 <= 1
 0 <= w_k10_GSK_s116 <= 1
 0 <= w_k10_GSK_s117 <= 1
 0 <= w_k10_AZN_s0 <= 1
 0 <= w_k10_AZN_s1 <= 1
 0 <= w_k10_AZN_s00 <= 1
 0 <= w_k10_AZN_s01 <= 1
 0 <= w_k10_AZN_s10 <= 1
 0 <= w_k10_AZN_s11 <= 1
 0 <= w_k10_AZN_s000 <= 1
 0 <= w_k10_AZN_s001 <= 1
 0 <= w_k10_AZN_s002 <= 1
 0 <= w_k10_AZN_s003 <= 1
 0 <= w_k10_AZN_s004 <= 1
 0 <= w_k10_AZN_s005 <= 1
 0 <= w_k10_AZN_s006 <= 1
 0 <= w_k10_AZN_s007 <= 1
 0 <= w_k10_AZN_s010 <= 1
 0 <= w_k10_AZN_s011 <= 1
 0 <= w_k10_AZN_s012 <= 1
 0 <= w_k10_AZN_s013 <= 1
 0 <= w_k10_AZN_s014 <= 1
 0 <= w_k10_AZN_s015 <= 1
 0 <= w_k10_AZN_s016 <= 1
 0 <= w_k10_AZN_s017 <= 1
 0 <= w_k10_AZN_s100 <= 1
 0 <= w_k10_AZN_s101 <= 1
 0 <= w_k10_AZN_s102 <= 1
 0 <= w_k10_AZN_s103 <= 1
 0 <= w_k10_AZN_s104 <= 1
 0 <= w_k10_AZN_s105 <= 1
 0 <= w_k10_AZN_s106 <= 1
 0 <= w_k10_AZN_s107 <= 1
 0 <= w_k10_AZN_s110 <= 1
 0 <= w_k10_AZN_s111 <= 1
 0 <= w_k10_AZN_s112 <= 1
 0 <= w_k10_AZN_s113 <= 1
 0 <= w_k10_AZN_s114 <= 1
 0 <= w_k10_AZN_s115 <= 1
 0 <= w_k10_AZN_s116 <= 1
 0 <= w_k10_AZN_s117 <= 1
 0 <= w_k10_RIO_s0 <= 1
 0 <= w_k10_RIO_s1 <= 1
 0 <= w_k10_RIO_s00 <= 1
 0 <= w_k10_RIO_s01 <= 1
 0 <= w_k10_RIO_s10 <= 1
 0 <= w_k10_RIO_s11 <= 1
 0 <= w_k10_RIO_s000 <= 1
 0 <= w_k10_RIO_s001 <= 1
 0 <= w_k10_RIO_s002 <= 1
 0 <= w_k10_RIO_s003 <= 1
 0 <= w_k10_RIO_s004 <= 1
 0 <= w_k10_RIO_s005 <= 1
 0 <= w_k10_RIO_s006 <= 1
 0 <= w_k10_RIO_s007 <= 1
 0 <= w_k10_RIO_s010 <= 1
 0 <= w_k10_RIO_s011 <= 1
 0 <= w_k10_RIO_s012 <= 1
 0 <= w_k10_RIO_s013 <= 1
 0 <= w_k10_RIO_s014 <= 1
 0 <= w_k10_RIO_s015 <= 1
 0 <= w_k10_RIO_s016 <= 1
 0 <= w_k10_RIO_s017 <= 1
 0 <= w_k10_RIO_s100 <= 1
 0 <= w_k10_RIO_s101 <= 1
 0 <= w_k10_RIO_s102 <= 1
 0 <= w_k10_RIO_s103 <= 1
 0 <= w_k10_RIO_s104 <= 1
 0 <= w_k10_RIO_s105 <= 1
 0 <= w_k10_RIO_s106 <= 1
 0 <= w_k10_RIO_s107 <= 1
 0 <= w_k10_RIO_s110 <= 1
 0 <= w_k10_RIO_s111 <= 1
 0 <= w_k10_RIO_s112 <= 1
 0 <= w_k10_RIO_s113 <= 1
 0 <= w_k10_RIO_s114 <= 1
 0 <= w_k10_RIO_s115 <= 1
 0 <= w_k10_RIO_s116 <= 1
 0 <= w_k10_RIO_s117 <= 1
 0 <= w_k10_BG_s0 <= 1
 0 <= w_k10_BG_s1 <= 1
 0 <= w_k10_BG_s00 <= 1
 0 <= w_k10_BG_s01 <= 1
 0 <= w_k10_BG_s10 <= 1
 0 <= w_k10_BG_s11 <= 1
 0 <= w_k10_BG_s000 <= 1
 0 <= w_k10_BG_s001 <= 1
 0 <= w_k10_BG_s002 <= 1
 0 <= w_k10_BG_s003 <= 1
 0 <= w_k10_BG_s004 <= 1
 0 <= w_k10_BG_s005 <= 1
 0 <= w_k10_BG_s006 <= 1
 0 <= w_k10_BG_s007 <= 1
 0 <= w_k10_BG_s010 <= 1
 0 <= w_k10_BG_s011 <= 1
 0 <= w_k10_BG_s012 <= 1
 0 <= w_k10_BG_s013 <= 1
 0 <= w_k10_BG_s014 <= 1
 0 <= w_k10_BG_s015 <= 1
 0 <= w_k10_BG_s016 <= 1
 0 <= w_k10_BG_s017 <= 1
 0 <= w_k10_BG_s100 <= 1
 0 <= w_k10_BG_s101 <= 1
 0 <= w_k10_BG_s102 <= 1
 0 <= w_k10_BG_s103 <= 1
 0 <= w_k10_BG_s104 <= 1
 0 <= w_k10_BG_s105 <= 1
 0 <= w_k10_BG_s106 <= 1
 0 <= w_k10_BG_s107 <= 1
 0 <= w_k10_BG_s110 <= 1
 0 <= w_k10_BG_s111 <= 1
 0 <= w_k10_BG_s112 <= 1
 0 <= w_k10_BG_s113 <= 1
 0 <= w_k10_BG_s114 <= 1
 0 <= w_k10_BG_s115 <= 1
 0 <= w_k10_BG_s116 <= 1
 0 <= w_k10_BG_s117 <= 1
 0 <= w_k10_TSCO_s0 <= 1
 0 <= w_k10_TSCO_s1 <= 1
 0 <= w_k10_TSCO_s00 <= 1
 0 <= w_k10_TSCO_s01 <= 1
 0 <= w_k10_TSCO_s10 <= 1
 0 <= w_k10_TSCO_s11 <= 1
 0 <= w_k10_TSCO_s000 <= 1
 0 <= w_k10_TSCO_s001 <= 1
 0 <= w_k10_TSCO_s002 <= 1
 0 <= w_k10_TSCO_s003 <= 1
 0 <= w_k10_TSCO_s004 <= 1
 0 <= w_k10_TSCO_s005 <= 1
 0 <= w_k10_TSCO_s006 <= 1
 0 <= w_k10_TSCO_s007 <= 1
 0 <= w_k10_TSCO_s010 <= 1
 0 <= w_k10_TSCO_s011 <= 1
 0 <= w_k10_TSCO_s012 <= 1
 0 <= w_k10_TSCO_s013 <= 1
 0 <= w_k10_TSCO_s014 <= 1
 0 <= w_k10_TSCO_s015 <= 1
 0 <= w_k10_TSCO_s016 <= 1
 0 <= w_k10_TSCO_s017 <= 1
 0 <= w_k10_TSCO_s100 <= 1
 0 <= w_k10_TSCO_s101 <= 1
 0 <= w_k10_TSCO_s102 <= 1
 0 <= w_k10_TSCO_s103 <= 1
 0 <= w_k10_TSCO_s104 <= 1
 0 <= w_k10_TSCO_s105 <= 1
 0 <= w_k10_TSCO_s106 <= 1
 0 <= w_k10_TSCO_s107 <= 1
 0 <= w_k10_TSCO_s110 <= 1
 0 <= w_k10_TSCO_s111 <= 1
 0 <= w_k10_TSCO_s112 <= 1
 0 <= w_k10_TSCO_s113 <= 1
 0 <= w_k10_TSCO_s114 <= 1
 0 <= w_k10_TSCO_s115 <= 1
 0 <= w_k10_TSCO_s116 <= 1
 0 <= w_k10_TSCO_s117 <= 1
 0 <= w_k10_AAL_s0 <= 1
 0 <= w_k10_AAL_s1 <= 1
 0 <= w_k10_AAL_s00 <= 1
 0 <= w_k10_AAL_s01 <= 1
 0 <= w_k10_AAL_s10 <= 1
 0 <= w_k10_AAL_s11 <= 1
 0 <= w_k10_AAL_s000 <= 1
 0 <= w_k10_AAL_s001 <= 1
 0 <= w_k10_AAL_s002 <= 1
 0 <= w_k10_AAL_s003 <= 1
 0 <= w_k10_AAL_s004 <= 1
 0 <= w_k10_AAL_s005 <= 1
 0 <= w_k10_AAL_s006 <= 1
 0 <= w_k10_AAL_s007 <= 1
 0 <= w_k10_AAL_s010 <= 1
 0 <= w_k10_AAL_s011 <= 1
 0 <= w_k10_AAL_s012 <= 1
 0 <= w_k10_AAL_s013 <= 1
 0 <= w_k10_AAL_s014 <= 1
 0 <= w_k10_AAL_s015 <= 1
 0 <= w_k10_AAL_s016 <= 1
 0 <= w_k10_AAL_s017 <= 1
 0 <= w_k10_AAL_s100 <= 1
 0 <= w_k10_AAL_s101 <= 1
 0 <= w_k10_AAL_s102 <= 1
 0 <= w_k10_AAL_s103 <= 1
 0 <= w_k10_AAL_s104 <= 1
 0 <= w_k10_AAL_s105 <= 1
 0 <= w_k10_AAL_s106 <= 1
 0 <= w_k10_AAL_s107 <= 1
 0 <= w_k10_AAL_s110 <= 1
 0 <= w_k10_AAL_s111 <= 1
 0 <= w_k10_AAL_s112 <= 1
 0 <= w_k10_AAL_s113 <= 1
 0 <= w_k10_AAL_s114 <= 1
 0 <= w_k10_AAL_s115 <= 1
 0 <= w_k10_AAL_s116 <= 1
 0 <= w_k10_AAL_s117 <= 1
 0 <= w_k10_PRU_s0 <= 1
 0 <= w_k10_PRU_s1 <= 1
 0 <= w_k10_PRU_s00 <= 1
 0 <= w_k10_PRU_s01 <= 1
 0 <= w_k10_PRU_s10 <= 1
 0 <= w_k10_PRU_s11 <= 1
 0 <= w_k10_PRU_s000 <= 1
 0 <= w_k10_PRU_s001 <= 1
 0 <= w_k10_PRU_s002 <= 1
 0 <= w_k10_PRU_s003 <= 1
 0 <= w_k10_PRU_s004 <= 1
 0 <= w_k10_PRU_s005 <= 1
 0 <= w_k10_PRU_s006 <= 1
 0 <= w_k10_PRU_s007 <= 1
 0 <= w_k10_PRU_s010 <= 1
 0 <= w_k10_PRU_s011 <= 1
 0 <= w_k10_PRU_s012 <= 1
 0 <= w_k10_PRU_s013 <= 1
 0 <= w_k10_PRU_s014 <= 1
 0 <= w_k10_PRU_s015 <= 1
 0 <= w_k10_PRU_s016 <= 1
 0 <= w_k10_PRU_s017 <= 1
 0 <= w_k10_PRU_s100 <= 1
 0 <= w_k10_PRU_s101 <= 1
 0 <= w_k10_PRU_s102 <= 1
 0 <= w_k10_PRU_s103 <= 1
 0 <= w_k10_PRU_s104 <= 1
 0 <= w_k10_PRU_s105 <= 1
 0 <= w_k10_PRU_s106 <= 1
 0 <= w_k10_PRU_s107 <= 1
 0 <= w_k10_PRU_s110 <= 1
 0 <= w_k10_PRU_s111 <= 1
 0 <= w_k10_PRU_s112 <= 1
 0 <= w_k10_PRU_s113 <= 1
 0 <= w_k10_PRU_s114 <= 1
 0 <= w_k10_PRU_s115 <= 1
 0 <= w_k10_PRU_s116 <= 1
 0 <= w_k10_PRU_s117 <= 1
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
 0 <= b_k10_HSBC_s00 <= 0.2
 0 <= b_k10_HSBC_s01 <= 0.2
 0 <= b_k10_HSBC_s10 <= 0.2
 0 <= b_k10_HSBC_s11 <= 0.2
 0 <= b_k10_HSBC_s000 <= 0.2
 0 <= b_k10_HSBC_s001 <= 0.2
 0 <= b_k10_HSBC_s002 <= 0.2
 0 <= b_k10_HSBC_s003 <= 0.2
 0 <= b_k10_HSBC_s004 <= 0.2
 0 <= b_k10_HSBC_s005 <= 0.2
 0 <= b_k10_HSBC_s006 <= 0.2
 0 <= b_k10_HSBC_s007 <= 0.2
 0 <= b_k10_HSBC_s010 <= 0.2
 0 <= b_k10_HSBC_s011 <= 0.2
 0 <= b_k10_HSBC_s012 <= 0.2
 0 <= b_k10_HSBC_s013 <= 0.2
 0 <= b_k10_HSBC_s014 <= 0.2
 0 <= b_k10_HSBC_s015 <= 0.2
 0 <= b_k10_HSBC_s016 <= 0.2
 0 <= b_k10_HSBC_s017 <= 0.2
 0 <= b_k10_HSBC_s100 <= 0.2
 0 <= b_k10_HSBC_s101 <= 0.2
 0 <= b_k10_HSBC_s102 <= 0.2
 0 <= b_k10_HSBC_s103 <= 0.2
 0 <= b_k10_HSBC_s104 <= 0.2
 0 <= b_k10_HSBC_s105 <= 0.2
 0 <= b_k10_HSBC_s106 <= 0.2
 0 <= b_k10_HSBC_s107 <= 0.2
 0 <= b_k10_HSBC_s110 <= 0.2
 0 <= b_k10_HSBC_s111 <= 0.2
 0 <= b_k10_HSBC_s112 <= 0.2
 0 <= b_k10_HSBC_s113 <= 0.2
 0 <= b_k10_HSBC_s114 <= 0.2
 0 <= b_k10_HSBC_s115 <= 0.2
 0 <= b_k10_HSBC_s116 <= 0.2
 0 <= b_k10_HSBC_s117 <= 0.2
 0 <= b_k10_VOD_s0 <= 0.2
 0 <= b_k10_VOD_s1 <= 0.2
 0 <= b_k10_VOD_s00 <= 0.2
 0 <= b_k10_VOD_s01 <= 0.2
 0 <= b_k10_VOD_s10 <= 0.2
 0 <= b_k10_VOD_s11 <= 0.2
 0 <= b_k10_VOD_s000 <= 0.2
 0 <= b_k10_VOD_s001 <= 0.2
 0 <= b_k10_VOD_s002 <= 0.2
 0 <= b_k10_VOD_s003 <= 0.2
 0 <= b_k10_VOD_s004 <= 0.2
 0 <= b_k10_VOD_s005 <= 0.2
 0 <= b_k10_VOD_s006 <= 0.2
 0 <= b_k10_VOD_s007 <= 0.2
 0 <= b_k10_VOD_s010 <= 0.2
 0 <= b_k10_VOD_s011 <= 0.2
 0 <= b_k10_VOD_s012 <= 0.2
 0 <= b_k10_VOD_s013 <= 0.2
 0 <= b_k10_VOD_s014 <= 0.2
 0 <= b_k10_VOD_s015 <= 0.2
 0 <= b_k10_VOD_s016 <= 0.2
 0 <= b_k10_VOD_s017 <= 0.2
 0 <= b_k10_VOD_s100 <= 0.2
 0 <= b_k10_VOD_s101 <= 0.2
 0 <= b_k10_VOD_s102 <= 0.2
 0 <= b_k10_VOD_s103 <= 0.2
 0 <= b_k10_VOD_s104 <= 0.2
 0 <= b_k10_VOD_s105 <= 0.2
 0 <= b_k10_VOD_s106 <= 0.2
 0 <= b_k10_VOD_s107 <= 0.2
 0 <= b_k10_VOD_s110 <= 0.2
 0 <= b_k10_VOD_s111 <= 0.2
 0 <= b_k10_VOD_s112 <= 0.2
 0 <= b_k10_VOD_s113 <= 0.2
 0 <= b_k10_VOD_s114 <= 0.2
 0 <= b_k10_VOD_s115 <= 0.2
 0 <= b_k10_VOD_s116 <= 0.2
 0 <= b_k10_VOD_s117 <= 0.2
 0 <= b_k10_BP_s0 <= 0.2
 0 <= b_k10_BP_s1 <= 0.2
 0 <= b_k10_BP_s00 <= 0.2
 0 <= b_k10_BP_s01 <= 0.2
 0 <= b_k10_BP_s10 <= 0.2
 0 <= b_k10_BP_s11 <= 0.2
 0 <= b_k10_BP_s000 <= 0.2
 0 <= b_k10_BP_s001 <= 0.2
 0 <= b_k10_BP_s002 <= 0.2
 0 <= b_k10_BP_s003 <= 0.2
 0 <= b_k10_BP_s004 <= 0.2
 0 <= b_k10_BP_s005 <= 0.2
 0 <= b_k10_BP_s006 <= 0.2
 0 <= b_k10_BP_s007 <= 0.2
 0 <= b_k10_BP_s010 <= 0.2
 0 <= b_k10_BP_s011 <= 0.2
 0 <= b_k10_BP_s012 <= 0.2
 0 <= b_k10_BP_s013 <= 0.2
 0 <= b_k10_BP_s014 <= 0.2
 0 <= b_k10_BP_s015 <= 0.2
 0 <= b_k10_BP_s016 <= 0.2
 0 <= b_k10_BP_s017 <= 0.2
 0 <= b_k10_BP_s100 <= 0.2
 0 <= b_k10_BP_s101 <= 0.2
 0 <= b_k10_BP_s102 <= 0.2
 0 <= b_k10_BP_s103 <= 0.2
 0 <= b_k10_BP_s104 <= 0.2
 0 <= b_k10_BP_s105 <= 0.2
 0 <= b_k10_BP_s106 <= 0.2
 0 <= b_k10_BP_s107 <= 0.2
 0 <= b_k10_BP_s110 <= 0.2
 0 <= b_k10_BP_s111 <= 0.2
 0 <= b_k10_BP_s112 <= 0.2
 0 <= b_k10_BP_s113 <= 0.2
 0 <= b_k10_BP_s114 <= 0.2
 0 <= b_k10_BP_s115 <= 0.2
 0 <= b_k10_BP_s116 <= 0.2
 0 <= b_k10_BP_s117 <= 0.2
 0 <= b_k10_GSK_s0 <= 0.2
 0 <= b_k10_GSK_s1 <= 0.2
 0 <= b_k10_GSK_s00 <= 0.2
 0 <= b_k10_GSK_s01 <= 0.2
 0 <= b_k10_GSK_s10 <= 0.2
 0 <= b_k10_GSK_s11 <= 0.2
 0 <= b_k10_GSK_s000 <= 0.2
 0 <= b_k10_GSK_s001 <= 0.2
 0 <= b_k10_GSK_s002 <= 0.2
 0 <= b_k10_GSK_s003 <= 0.2
 0 <= b_k10_GSK_s004 <= 0.2
 0 <= b_k10_GSK_s005 <= 0.2
 0 <= b_k10_GSK_s006 <= 0.2
 0 <= b_k10_GSK_s007 <= 0.2
 0 <= b_k10_GSK_s010 <= 0.2
 0 <= b_k10_GSK_s011 <= 0.2
 0 <= b_k10_GSK_s012 <= 0.2
 0 <= b_k10_GSK_s013 <= 0.2
 0 <= b_k10_GSK_s014 <= 0.2
 0 <= b_k10_GSK_s015 <= 0.2
 0 <= b_k10_GSK_s016 <= 0.2
 0 <= b_k10_GSK_s017 <= 0.2
 0 <= b_k10_GSK_s100 <= 0.2
 0 <= b_k10_GSK_s101 <= 0.2
 0 <= b_k10_GSK_s102 <= 0.2
 0 <= b_k10_GSK_s103 <= 0.2
 0 <= b_k10_GSK_s104 <= 0.2
 0 <= b_k10_GSK_s105 <= 0.2
 0 <= b_k10_GSK_s106 <= 0.2
 0 <= b_k10_GSK_s107 <= 0.2
 0 <= b_k10_GSK_s110 <= 0.2
 0 <= b_k10_GSK_s111 <= 0.2
 0 <= b_k10_GSK_s112 <= 0.2
 0 <= b_k10_GSK_s113 <= 0.2
 0 <= b_k10_GSK_s114 <= 0.2
 0 <= b_k10_GSK_s115 <= 0.2
 0 <= b_k10_GSK_s116 <= 0.2
 0 <= b_k10_GSK_s117 <= 0.2
 0 <= b_k10_AZN_s0 <= 0.2
 0 <= b_k10_AZN_s1 <= 0.2
 0 <= b_k10_AZN_s00 <= 0.2
 0 <= b_k10_AZN_s01 <= 0.2
 0 <= b_k10_AZN_s10 <= 0.2
 0 <= b_k10_AZN_s11 <= 0.2
 0 <= b_k10_AZN_s000 <= 0.2
 0 <= b_k10_AZN_s001 <= 0.2
 0 <= b_k10_AZN_s002 <= 0.2
 0 <= b_k10_AZN_s003 <= 0.2
 0 <= b_k10_AZN_s004 <= 0.2
 0 <= b_k10_AZN_s005 <= 0.2
 0 <= b_k10_AZN_s006 <= 0.2
 0 <= b_k10_AZN_s007 <= 0.2
 0 <= b_k10_AZN_s010 <= 0.2
 0 <= b_k10_AZN_s011 <= 0.2
 0 <= b_k10_AZN_s012 <= 0.2
 0 <= b_k10_AZN_s013 <= 0.2
 0 <= b_k10_AZN_s014 <= 0.2
 0 <= b_k10_AZN_s015 <= 0.2
 0 <= b_k10_AZN_s016 <= 0.2
 0 <= b_k10_AZN_s017 <= 0.2
 0 <= b_k10_AZN_s100 <= 0.2
 0 <= b_k10_AZN_s101 <= 0.2
 0 <= b_k10_AZN_s102 <= 0.2
 0 <= b_k10_AZN_s103 <= 0.2
 0 <= b_k10_AZN_s104 <= 0.2
 0 <= b_k10_AZN_s105 <= 0.2
 0 <= b_k10_AZN_s106 <= 0.2
 0 <= b_k10_AZN_s107 <= 0.2
 0 <= b_k10_AZN_s110 <= 0.2
 0 <= b_k10_AZN_s111 <= 0.2
 0 <= b_k10_AZN_s112 <= 0.2
 0 <= b_k10_AZN_s113 <= 0.2
 0 <= b_k10_AZN_s114 <= 0.2
 0 <= b_k10_AZN_s115 <= 0.2
 0 <= b_k10_AZN_s116 <= 0.2
 0 <= b_k10_AZN_s117 <= 0.2
 0 <= b_k10_RIO_s0 <= 0.2
 0 <= b_k10_RIO_s1 <= 0.2
 0 <= b_k10_RIO_s00 <= 0.2
 0 <= b_k10_RIO_s01 <= 0.2
 0 <= b_k10_RIO_s10 <= 0.2
 0 <= b_k10_RIO_s11 <= 0.2
 0 <= b_k10_RIO_s000 <= 0.2
 0 <= b_k10_RIO_s001 <= 0.2
 0 <= b_k10_RIO_s002 <= 0.2
 0 <= b_k10_RIO_s003 <= 0.2
 0 <= b_k10_RIO_s004 <= 0.2
 0 <= b_k10_RIO_s005 <= 0.2
 0 <= b_k10_RIO_s006 <= 0.2
 0 <= b_k10_RIO_s007 <= 0.2
 0 <= b_k10_RIO_s010 <= 0.2
 0 <= b_k10_RIO_s011 <= 0.2
 0 <= b_k10_RIO_s012 <= 0.2
 0 <= b_k10_RIO_s013 <= 0.2
 0 <= b_k10_RIO_s014 <= 0.2
 0 <= b_k10_RIO_s015 <= 0.2
 0 <= b_k10_RIO_s016 <= 0.2
 0 <= b_k10_RIO_s017 <= 0.2
 0 <= b_k10_RIO_s100 <= 0.2
 0 <= b_k10_RIO_s101 <= 0.2
 0 <= b_k10_RIO_s102 <= 0.2
 0 <= b_k10_RIO_s103 <= 0.2
 0 <= b_k10_RIO_s104 <= 0.2
 0 <= b_k10_RIO_s105 <= 0.2
 0 <= b_k10_RIO_s106 <= 0.2
 0 <= b_k10_RIO_s107 <= 0.2
 0 <= b_k10_RIO_s110 <= 0.2
 0 <= b_k10_RIO_s111 <= 0.2
 0 <= b_k10_RIO_s112 <= 0.2
 0 <= b_k10_RIO_s113 <= 0.2
 0 <= b_k10_RIO_s114 <= 0.2
 0 <= b_k10_RIO_s115 <= 0.2
 0 <= b_k10_RIO_s116 <= 0.2
 0 <= b_k10_RIO_s117 <= 0.2
 0 <= b_k10_BG_s0 <= 0.2
 0 <= b_k10_BG_s1 <= 0.2
 0 <= b_k10_BG_s00 <= 0.2
 0 <= b_k10_BG_s01 <= 0.2
 0 <= b_k10_BG_s10 <= 0.2
 0 <= b_k10_BG_s11 <= 0.2
 0 <= b_k10_BG_s000 <= 0.2
 0 <= b_k10_BG_s001 <= 0.2
 0 <= b_k10_BG_s002 <= 0.2
 0 <= b_k10_BG_s003 <= 0.2
 0 <= b_k10_BG_s004 <= 0.2
 0 <= b_k10_BG_s005 <= 0.2
 0 <= b_k10_BG_s006 <= 0.2
 0 <= b_k10_BG_s007 <= 0.2
 0 <= b_k10_BG_s010 <= 0.2
 0 <= b_k10_BG_s011 <= 0.2
 0 <= b_k10_BG_s012 <= 0.2
 0 <= b_k10_BG_s013 <= 0.2
 0 <= b_k10_BG_s014 <= 0.2
 0 <= b_k10_BG_s015 <= 0.2
 0 <= b_k10_BG_s016 <= 0.2
 0 <= b_k10_BG_s017 <= 0.2
 0 <= b_k10_BG_s100 <= 0.2
 0 <= b_k10_BG_s101 <= 0.2
 0 <= b_k10_BG_s102 <= 0.2
 0 <= b_k10_BG_s103 <= 0.2
 0 <= b_k10_BG_s104 <= 0.2
 0 <= b_k10_BG_s105 <= 0.2
 0 <= b_k10_BG_s106 <= 0.2
 0 <= b_k10_BG_s107 <= 0.2
 0 <= b_k10_BG_s110 <= 0.2
 0 <= b_k10_BG_s111 <= 0.2
 0 <= b_k10_BG_s112 <= 0.2
 0 <= b_k10_BG_s113 <= 0.2
 0 <= b_k10_BG_s114 <= 0.2
 0 <= b_k10_BG_s115 <= 0.2
 0 <= b_k10_BG_s116 <= 0.2
 0 <= b_k10_BG_s117 <= 0.2
 0 <= b_k10_TSCO_s0 <= 0.2
 0 <= b_k10_TSCO_s1 <= 0.2
 0 <= b_k10_TSCO_s00 <= 0.2
 0 <= b_k10_TSCO_s01 <= 0.2
 0 <= b_k10_TSCO_s10 <= 0.2
 0 <= b_k10_TSCO_s11 <= 0.2
 0 <= b_k10_TSCO_s000 <= 0.2
 0 <= b_k10_TSCO_s001 <= 0.2
 0 <= b_k10_TSCO_s002 <= 0.2
 0 <= b_k10_TSCO_s003 <= 0.2
 0 <= b_k10_TSCO_s004 <= 0.2
 0 <= b_k10_TSCO_s005 <= 0.2
 0 <= b_k10_TSCO_s006 <= 0.2
 0 <= b_k10_TSCO_s007 <= 0.2
 0 <= b_k10_TSCO_s010 <= 0.2
 0 <= b_k10_TSCO_s011 <= 0.2
 0 <= b_k10_TSCO_s012 <= 0.2
 0 <= b_k10_TSCO_s013 <= 0.2
 0 <= b_k10_TSCO_s014 <= 0.2
 0 <= b_k10_TSCO_s015 <= 0.2
 0 <= b_k10_TSCO_s016 <= 0.2
 0 <= b_k10_TSCO_s017 <= 0.2
 0 <= b_k10_TSCO_s100 <= 0.2
 0 <= b_k10_TSCO_s101 <= 0.2
 0 <= b_k10_TSCO_s102 <= 0.2
 0 <= b_k10_TSCO_s103 <= 0.2
 0 <= b_k10_TSCO_s104 <= 0.2
 0 <= b_k10_TSCO_s105 <= 0.2
 0 <= b_k10_TSCO_s106 <= 0.2
 0 <= b_k10_TSCO_s107 <= 0.2
 0 <= b_k10_TSCO_s110 <= 0.2
 0 <= b_k10_TSCO_s111 <= 0.2
 0 <= b_k10_TSCO_s112 <= 0.2
 0 <= b_k10_TSCO_s113 <= 0.2
 0 <= b_k10_TSCO_s114 <= 0.2
 0 <= b_k10_TSCO_s115 <= 0.2
 0 <= b_k10_TSCO_s116 <= 0.2
 0 <= b_k10_TSCO_s117 <= 0.2
 0 <= b_k10_AAL_s0 <= 0.2
 0 <= b_k10_AAL_s1 <= 0.2
 0 <= b_k10_AAL_s00 <= 0.2
 0 <= b_k10_AAL_s01 <= 0.2
 0 <= b_k10_AAL_s10 <= 0.2
 0 <= b_k10_AAL_s11 <= 0.2
 0 <= b_k10_AAL_s000 <= 0.2
 0 <= b_k10_AAL_s001 <= 0.2
 0 <= b_k10_AAL_s002 <= 0.2
 0 <= b_k10_AAL_s003 <= 0.2
 0 <= b_k10_AAL_s004 <= 0.2
 0 <= b_k10_AAL_s005 <= 0.2
 0 <= b_k10_AAL_s006 <= 0.2
 0 <= b_k10_AAL_s007 <= 0.2
 0 <= b_k10_AAL_s010 <= 0.2
 0 <= b_k10_AAL_s011 <= 0.2
 0 <= b_k10_AAL_s012 <= 0.2
 0 <= b_k10_AAL_s013 <= 0.2
 0 <= b_k10_AAL_s014 <= 0.2
 0 <= b_k10_AAL_s015 <= 0.2
 0 <= b_k10_AAL_s016 <= 0.2
 0 <= b_k10_AAL_s017 <= 0.2
 0 <= b_k10_AAL_s100 <= 0.2
 0 <= b_k10_AAL_s101 <= 0.2
 0 <= b_k10_AAL_s102 <= 0.2
 0 <= b_k10_AAL_s103 <= 0.2
 0 <= b_k10_AAL_s104 <= 0.2
 0 <= b_k10_AAL_s105 <= 0.2
 0 <= b_k10_AAL_s106 <= 0.2
 0 <= b_k10_AAL_s107 <= 0.2
 0 <= b_k10_AAL_s110 <= 0.2
 0 <= b_k10_AAL_s111 <= 0.2
 0 <= b_k10_AAL_s112 <= 0.2
 0 <= b_k10_AAL_s113 <= 0.2
 0 <= b_k10_AAL_s114 <= 0.2
 0 <= b_k10_AAL_s115 <= 0.2
 0 <= b_k10_AAL_s116 <= 0.2
 0 <= b_k10_AAL_s117 <= 0.2
 0 <= b_k10_PRU_s0 <= 0.2
 0 <= b_k10_PRU_s1 <= 0.2
 0 <= b_k10_PRU_s00 <= 0.2
 0 <= b_k10_PRU_s01 <= 0.2
 0 <= b_k10_PRU_s10 <= 0.2
 0 <= b_k10_PRU_s11 <= 0.2
 0 <= b_k10_PRU_s000 <= 0.2
 0 <= b_k10_PRU_s001 <= 0.2
 0 <= b_k10_PRU_s002 <= 0.2
 0 <= b_k10_PRU_s003 <= 0.2
 0 <= b_k10_PRU_s004 <= 0.2
 0 <= b_k10_PRU_s005 <= 0.2
 0 <= b_k10_PRU_s006 <= 0.2
 0 <= b_k10_PRU_s007 <= 0.2
 0 <= b_k10_PRU_s010 <= 0.2
 0 <= b_k10_PRU_s011 <= 0.2
 0 <= b_k10_PRU_s012 <= 0.2
 0 <= b_k10_PRU_s013 <= 0.2
 0 <= b_k10_PRU_s014 <= 0.2
 0 <= b_k10_PRU_s015 <= 0.2
 0 <= b_k10_PRU_s016 <= 0.2
 0 <= b_k10_PRU_s017 <= 0.2
 0 <= b_k10_PRU_s100 <= 0.2
 0 <= b_k10_PRU_s101 <= 0.2
 0 <= b_k10_PRU_s102 <= 0.2
 0 <= b_k10_PRU_s103 <= 0.2
 0 <= b_k10_PRU_s104 <= 0.2
 0 <= b_k10_PRU_s105 <= 0.2
 0 <= b_k10_PRU_s106 <= 0.2
 0 <= b_k10_PRU_s107 <= 0.2
 0 <= b_k10_PRU_s110 <= 0.2
 0 <= b_k10_PRU_s111 <= 0.2
 0 <= b_k10_PRU_s112 <= 0.2
 0 <= b_k10_PRU_s113 <= 0.2
 0 <= b_k10_PRU_s114 <= 0.2
 0 <= b_k10_PRU_s115 <= 0.2
 0 <= b_k10_PRU_s116 <= 0.2
 0 <= b_k10_PRU_s117 <= 0.2
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
 0 <= s_k10_HSBC_s00 <= 0.2
 0 <= s_k10_HSBC_s01 <= 0.2
 0 <= s_k10_HSBC_s10 <= 0.2
 0 <= s_k10_HSBC_s11 <= 0.2
 0 <= s_k10_HSBC_s000 <= 0.2
 0 <= s_k10_HSBC_s001 <= 0.2
 0 <= s_k10_HSBC_s002 <= 0.2
 0 <= s_k10_HSBC_s003 <= 0.2
 0 <= s_k10_HSBC_s004 <= 0.2
 0 <= s_k10_HSBC_s005 <= 0.2
 0 <= s_k10_HSBC_s006 <= 0.2
 0 <= s_k10_HSBC_s007 <= 0.2
 0 <= s_k10_HSBC_s010 <= 0.2
 0 <= s_k10_HSBC_s011 <= 0.2
 0 <= s_k10_HSBC_s012 <= 0.2
 0 <= s_k10_HSBC_s013 <= 0.2
 0 <= s_k10_HSBC_s014 <= 0.2
 0 <= s_k10_HSBC_s015 <= 0.2
 0 <= s_k10_HSBC_s016 <= 0.2
 0 <= s_k10_HSBC_s017 <= 0.2
 0 <= s_k10_HSBC_s100 <= 0.2
 0 <= s_k10_HSBC_s101 <= 0.2
 0 <= s_k10_HSBC_s102 <= 0.2
 0 <= s_k10_HSBC_s103 <= 0.2
 0 <= s_k10_HSBC_s104 <= 0.2
 0 <= s_k10_HSBC_s105 <= 0.2
 0 <= s_k10_HSBC_s106 <= 0.2
 0 <= s_k10_HSBC_s107 <= 0.2
 0 <= s_k10_HSBC_s110 <= 0.2
 0 <= s_k10_HSBC_s111 <= 0.2
 0 <= s_k10_HSBC_s112 <= 0.2
 0 <= s_k10_HSBC_s113 <= 0.2
 0 <= s_k10_HSBC_s114 <= 0.2
 0 <= s_k10_HSBC_s115 <= 0.2
 0 <= s_k10_HSBC_s116 <= 0.2
 0 <= s_k10_HSBC_s117 <= 0.2
 0 <= s_k10_VOD_s0 <= 0.2
 0 <= s_k10_VOD_s1 <= 0.2
 0 <= s_k10_VOD_s00 <= 0.2
 0 <= s_k10_VOD_s01 <= 0.2
 0 <= s_k10_VOD_s10 <= 0.2
 0 <= s_k10_VOD_s11 <= 0.2
 0 <= s_k10_VOD_s000 <= 0.2
 0 <= s_k10_VOD_s001 <= 0.2
 0 <= s_k10_VOD_s002 <= 0.2
 0 <= s_k10_VOD_s003 <= 0.2
 0 <= s_k10_VOD_s004 <= 0.2
 0 <= s_k10_VOD_s005 <= 0.2
 0 <= s_k10_VOD_s006 <= 0.2
 0 <= s_k10_VOD_s007 <= 0.2
 0 <= s_k10_VOD_s010 <= 0.2
 0 <= s_k10_VOD_s011 <= 0.2
 0 <= s_k10_VOD_s012 <= 0.2
 0 <= s_k10_VOD_s013 <= 0.2
 0 <= s_k10_VOD_s014 <= 0.2
 0 <= s_k10_VOD_s015 <= 0.2
 0 <= s_k10_VOD_s016 <= 0.2
 0 <= s_k10_VOD_s017 <= 0.2
 0 <= s_k10_VOD_s100 <= 0.2
 0 <= s_k10_VOD_s101 <= 0.2
 0 <= s_k10_VOD_s102 <= 0.2
 0 <= s_k10_VOD_s103 <= 0.2
 0 <= s_k10_VOD_s104 <= 0.2
 0 <= s_k10_VOD_s105 <= 0.2
 0 <= s_k10_VOD_s106 <= 0.2
 0 <= s_k10_VOD_s107 <= 0.2
 0 <= s_k10_VOD_s110 <= 0.2
 0 <= s_k10_VOD_s111 <= 0.2
 0 <= s_k10_VOD_s112 <= 0.2
 0 <= s_k10_VOD_s113 <= 0.2
 0 <= s_k10_VOD_s114 <= 0.2
 0 <= s_k10_VOD_s115 <= 0.2
 0 <= s_k10_VOD_s116 <= 0.2
 0 <= s_k10_VOD_s117 <= 0.2
 0 <= s_k10_BP_s0 <= 0.2
 0 <= s_k10_BP_s1 <= 0.2
 0 <= s_k10_BP_s00 <= 0.2
 0 <= s_k10_BP_s01 <= 0.2
 0 <= s_k10_BP_s10 <= 0.2
 0 <= s_k10_BP_s11 <= 0.2
 0 <= s_k10_BP_s000 <= 0.2
 0 <= s_k10_BP_s001 <= 0.2
 0 <= s_k10_BP_s002 <= 0.2
 0 <= s_k10_BP_s003 <= 0.2
 0 <= s_k10_BP_s004 <= 0.2
 0 <= s_k10_BP_s005 <= 0.2
 0 <= s_k10_BP_s006 <= 0.2
 0 <= s_k10_BP_s007 <= 0.2
 0 <= s_k10_BP_s010 <= 0.2
 0 <= s_k10_BP_s011 <= 0.2
 0 <= s_k10_BP_s012 <= 0.2
 0 <= s_k10_BP_s013 <= 0.2
 0 <= s_k10_BP_s014 <= 0.2
 0 <= s_k10_BP_s015 <= 0.2
 0 <= s_k10_BP_s016 <= 0.2
 0 <= s_k10_BP_s017 <= 0.2
 0 <= s_k10_BP_s100 <= 0.2
 0 <= s_k10_BP_s101 <= 0.2
 0 <= s_k10_BP_s102 <= 0.2
 0 <= s_k10_BP_s103 <= 0.2
 0 <= s_k10_BP_s104 <= 0.2
 0 <= s_k10_BP_s105 <= 0.2
 0 <= s_k10_BP_s106 <= 0.2
 0 <= s_k10_BP_s107 <= 0.2
 0 <= s_k10_BP_s110 <= 0.2
 0 <= s_k10_BP_s111 <= 0.2
 0 <= s_k10_BP_s112 <= 0.2
 0 <= s_k10_BP_s113 <= 0.2
 0 <= s_k10_BP_s114 <= 0.2
 0 <= s_k10_BP_s115 <= 0.2
 0 <= s_k10_BP_s116 <= 0.2
 0 <= s_k10_BP_s117 <= 0.2
 0 <= s_k10_GSK_s0 <= 0.2
 0 <= s_k10_GSK_s1 <= 0.2
 0 <= s_k10_GSK_s00 <= 0.2
 0 <= s_k10_GSK_s01 <= 0.2
 0 <= s_k10_GSK_s10 <= 0.2
 0 <= s_k10_GSK_s11 <= 0.2
 0 <= s_k10_GSK_s000 <= 0.2
 0 <= s_k10_GSK_s001 <= 0.2
 0 <= s_k10_GSK_s002 <= 0.2
 0 <= s_k10_GSK_s003 <= 0.2
 0 <= s_k10_GSK_s004 <= 0.2
 0 <= s_k10_GSK_s005 <= 0.2
 0 <= s_k10_GSK_s006 <= 0.2
 0 <= s_k10_GSK_s007 <= 0.2
 0 <= s_k10_GSK_s010 <= 0.2
 0 <= s_k10_GSK_s011 <= 0.2
 0 <= s_k10_GSK_s012 <= 0.2
 0 <= s_k10_GSK_s013 <= 0.2
 0 <= s_k10_GSK_s014 <= 0.2
 0 <= s_k10_GSK_s015 <= 0.2
 0 <= s_k10_GSK_s016 <= 0.2
 0 <= s_k10_GSK_s017 <= 0.2
 0 <= s_k10_GSK_s100 <= 0.2
 0 <= s_k10_GSK_s101 <= 0.2
 0 <= s_k10_GSK_s102 <= 0.2
 0 <= s_k10_GSK_s103 <= 0.2
 0 <= s_k10_GSK_s104 <= 0.2
 0 <= s_k10_GSK_s105 <= 0.2
 0 <= s_k10_GSK_s106 <= 0.2
 0 <= s_k10_GSK_s107 <= 0.2
 0 <= s_k10_GSK_s110 <= 0.2
 0 <= s_k10_GSK_s111 <= 0.2
 0 <= s_k10_GSK_s112 <= 0.2
 0 <= s_k10_GSK_s113 <= 0.2
 0 <= s_k10_GSK_s114 <= 0.2
 0 <= s_k10_GSK_s115 <= 0.2
 0 <= s_k10_GSK_s116 <= 0.2
 0 <= s_k10_GSK_s117 <= 0.2
 0 <= s_k10_AZN_s0 <= 0.2
 0 <= s_k10_AZN_s1 <= 0.2
 0 <= s_k10_AZN_s00 <= 0.2
 0 <= s_k10_AZN_s01 <= 0.2
 0 <= s_k10_AZN_s10 <= 0.2
 0 <= s_k10_AZN_s11 <= 0.2
 0 <= s_k10_AZN_s000 <= 0.2
 0 <= s_k10_AZN_s001 <= 0.2
 0 <= s_k10_AZN_s002 <= 0.2
 0 <= s_k10_AZN_s003 <= 0.2
 0 <= s_k10_AZN_s004 <= 0.2
 0 <= s_k10_AZN_s005 <= 0.2
 0 <= s_k10_AZN_s006 <= 0.2
 0 <= s_k10_AZN_s007 <= 0.2
 0 <= s_k10_AZN_s010 <= 0.2
 0 <= s_k10_AZN_s011 <= 0.2
 0 <= s_k10_AZN_s012 <= 0.2
 0 <= s_k10_AZN_s013 <= 0.2
 0 <= s_k10_AZN_s014 <= 0.2
 0 <= s_k10_AZN_s015 <= 0.2
 0 <= s_k10_AZN_s016 <= 0.2
 0 <= s_k10_AZN_s017 <= 0.2
 0 <= s_k10_AZN_s100 <= 0.2
 0 <= s_k10_AZN_s101 <= 0.2
 0 <= s_k10_AZN_s102 <= 0.2
 0 <= s_k10_AZN_s103 <= 0.2
 0 <= s_k10_AZN_s104 <= 0.2
 0 <= s_k10_AZN_s105 <= 0.2
 0 <= s_k10_AZN_s106 <= 0.2
 0 <= s_k10_AZN_s107 <= 0.2
 0 <= s_k10_AZN_s110 <= 0.2
 0 <= s_k10_AZN_s111 <= 0.2
 0 <= s_k10_AZN_s112 <= 0.2
 0 <= s_k10_AZN_s113 <= 0.2
 0 <= s_k10_AZN_s114 <= 0.2
 0 <= s_k10_AZN_s115 <= 0.2
 0 <= s_k10_AZN_s116 <= 0.2
 0 <= s_k10_AZN_s117 <= 0.2
 0 <= s_k10_RIO_s0 <= 0.2
 0 <= s_k10_RIO_s1 <= 0.2
 0 <= s_k10_RIO_s00 <= 0.2
 0 <= s_k10_RIO_s01 <= 0.2
 0 <= s_k10_RIO_s10 <= 0.2
 0 <= s_k10_RIO_s11 <= 0.2
 0 <= s_k10_RIO_s000 <= 0.2
 0 <= s_k10_RIO_s001 <= 0.2
 0 <= s_k10_RIO_s002 <= 0.2
 0 <= s_k10_RIO_s003 <= 0.2
 0 <= s_k10_RIO_s004 <= 0.2
 0 <= s_k10_RIO_s005 <= 0.2
 0 <= s_k10_RIO_s006 <= 0.2
 0 <= s_k10_RIO_s007 <= 0.2
 0 <= s_k10_RIO_s010 <= 0.2
 0 <= s_k10_RIO_s011 <= 0.2
 0 <= s_k10_RIO_s012 <= 0.2
 0 <= s_k10_RIO_s013 <= 0.2
 0 <= s_k10_RIO_s014 <= 0.2
 0 <= s_k10_RIO_s015 <= 0.2
 0 <= s_k10_RIO_s016 <= 0.2
 0 <= s_k10_RIO_s017 <= 0.2
 0 <= s_k10_RIO_s100 <= 0.2
 0 <= s_k10_RIO_s101 <= 0.2
 0 <= s_k10_RIO_s102 <= 0.2
 0 <= s_k10_RIO_s103 <= 0.2
 0 <= s_k10_RIO_s104 <= 0.2
 0 <= s_k10_RIO_s105 <= 0.2
 0 <= s_k10_RIO_s106 <= 0.2
 0 <= s_k10_RIO_s107 <= 0.2
 0 <= s_k10_RIO_s110 <= 0.2
 0 <= s_k10_RIO_s111 <= 0.2
 0 <= s_k10_RIO_s112 <= 0.2
 0 <= s_k10_RIO_s113 <= 0.2
 0 <= s_k10_RIO_s114 <= 0.2
 0 <= s_k10_RIO_s115 <= 0.2
 0 <= s_k10_RIO_s116 <= 0.2
 0 <= s_k10_RIO_s117 <= 0.2
 0 <= s_k10_BG_s0 <= 0.2
 0 <= s_k10_BG_s1 <= 0.2
 0 <= s_k10_BG_s00 <= 0.2
 0 <= s_k10_BG_s01 <= 0.2
 0 <= s_k10_BG_s10 <= 0.2
 0 <= s_k10_BG_s11 <= 0.2
 0 <= s_k10_BG_s000 <= 0.2
 0 <= s_k10_BG_s001 <= 0.2
 0 <= s_k10_BG_s002 <= 0.2
 0 <= s_k10_BG_s003 <= 0.2
 0 <= s_k10_BG_s004 <= 0.2
 0 <= s_k10_BG_s005 <= 0.2
 0 <= s_k10_BG_s006 <= 0.2
 0 <= s_k10_BG_s007 <= 0.2
 0 <= s_k10_BG_s010 <= 0.2
 0 <= s_k10_BG_s011 <= 0.2
 0 <= s_k10_BG_s012 <= 0.2
 0 <= s_k10_BG_s013 <= 0.2
 0 <= s_k10_BG_s014 <= 0.2
 0 <= s_k10_BG_s015 <= 0.2
 0 <= s_k10_BG_s016 <= 0.2
 0 <= s_k10_BG_s017 <= 0.2
 0 <= s_k10_BG_s100 <= 0.2
 0 <= s_k10_BG_s101 <= 0.2
 0 <= s_k10_BG_s102 <= 0.2
 0 <= s_k10_BG_s103 <= 0.2
 0 <= s_k10_BG_s104 <= 0.2
 0 <= s_k10_BG_s105 <= 0.2
 0 <= s_k10_BG_s106 <= 0.2
 0 <= s_k10_BG_s107 <= 0.2
 0 <= s_k10_BG_s110 <= 0.2
 0 <= s_k10_BG_s111 <= 0.2
 0 <= s_k10_BG_s112 <= 0.2
 0 <= s_k10_BG_s113 <= 0.2
 0 <= s_k10_BG_s114 <= 0.2
 0 <= s_k10_BG_s115 <= 0.2
 0 <= s_k10_BG_s116 <= 0.2
 0 <= s_k10_BG_s117 <= 0.2
 0 <= s_k10_TSCO_s0 <= 0.2
 0 <= s_k10_TSCO_s1 <= 0.2
 0 <= s_k10_TSCO_s00 <= 0.2
 0 <= s_k10_TSCO_s01 <= 0.2
 0 <= s_k10_TSCO_s10 <= 0.2
 0 <= s_k10_TSCO_s11 <= 0.2
 0 <= s_k10_TSCO_s000 <= 0.2
 0 <= s_k10_TSCO_s001 <= 0.2
 0 <= s_k10_TSCO_s002 <= 0.2
 0 <= s_k10_TSCO_s003 <= 0.2
 0 <= s_k10_TSCO_s004 <= 0.2
 0 <= s_k10_TSCO_s005 <= 0.2
 0 <= s_k10_TSCO_s006 <= 0.2
 0 <= s_k10_TSCO_s007 <= 0.2
 0 <= s_k10_TSCO_s010 <= 0.2
 0 <= s_k10_TSCO_s011 <= 0.2
 0 <= s_k10_TSCO_s012 <= 0.2
 0 <= s_k10_TSCO_s013 <= 0.2
 0 <= s_k10_TSCO_s014 <= 0.2
 0 <= s_k10_TSCO_s015 <= 0.2
 0 <= s_k10_TSCO_s016 <= 0.2
 0 <= s_k10_TSCO_s017 <= 0.2
 0 <= s_k10_TSCO_s100 <= 0.2
 0 <= s_k10_TSCO_s101 <= 0.2
 0 <= s_k10_TSCO_s102 <= 0.2
 0 <= s_k10_TSCO_s103 <= 0.2
 0 <= s_k10_TSCO_s104 <= 0.2
 0 <= s_k10_TSCO_s105 <= 0.2
 0 <= s_k10_TSCO_s106 <= 0.2
 0 <= s_k10_TSCO_s107 <= 0.2
 0 <= s_k10_TSCO_s110 <= 0.2
 0 <= s_k10_TSCO_s111 <= 0.2
 0 <= s_k10_TSCO_s112 <= 0.2
 0 <= s_k10_TSCO_s113 <= 0.2
 0 <= s_k10_TSCO_s114 <= 0.2
 0 <= s_k10_TSCO_s115 <= 0.2
 0 <= s_k10_TSCO_s116 <= 0.2
 0 <= s_k10_TSCO_s117 <= 0.2
 0 <= s_k10_AAL_s0 <= 0.2
 0 <= s_k10_AAL_s1 <= 0.2
 0 <= s_k10_AAL_s00 <= 0.2
 0 <= s_k10_AAL_s01 <= 0.2
 0 <= s_k10_AAL_s10 <= 0.2
 0 <= s_k10_AAL_s11 <= 0.2
 0 <= s_k10_AAL_s000 <= 0.2
 0 <= s_k10_AAL_s001 <= 0.2
 0 <= s_k10_AAL_s002 <= 0.2
 0 <= s_k10_AAL_s003 <= 0.2
 0 <= s_k10_AAL_s004 <= 0.2
 0 <= s_k10_AAL_s005 <= 0.2
 0 <= s_k10_AAL_s006 <= 0.2
 0 <= s_k10_AAL_s007 <= 0.2
 0 <= s_k10_AAL_s010 <= 0.2
 0 <= s_k10_AAL_s011 <= 0.2
 0 <= s_k10_AAL_s012 <= 0.2
 0 <= s_k10_AAL_s013 <= 0.2
 0 <= s_k10_AAL_s014 <= 0.2
 0 <= s_k10_AAL_s015 <= 0.2
 0 <= s_k10_AAL_s016 <= 0.2
 0 <= s_k10_AAL_s017 <= 0.2
 0 <= s_k10_AAL_s100 <= 0.2
 0 <= s_k10_AAL_s101 <= 0.2
 0 <= s_k10_AAL_s102 <= 0.2
 0 <= s_k10_AAL_s103 <= 0.2
 0 <= s_k10_AAL_s104 <= 0.2
 0 <= s_k10_AAL_s105 <= 0.2
 0 <= s_k10_AAL_s106 <= 0.2
 0 <= s_k10_AAL_s107 <= 0.2
 0 <= s_k10_AAL_s110 <= 0.2
 0 <= s_k10_AAL_s111 <= 0.2
 0 <= s_k10_AAL_s112 <= 0.2
 0 <= s_k10_AAL_s113 <= 0.2
 0 <= s_k10_AAL_s114 <= 0.2
 0 <= s_k10_AAL_s115 <= 0.2
 0 <= s_k10_AAL_s116 <= 0.2
 0 <= s_k10_AAL_s117 <= 0.2
 0 <= s_k10_PRU_s0 <= 0.2
 0 <= s_k10_PRU_s1 <= 0.2
 0 <= s_k10_PRU_s00 <= 0.2
 0 <= s_k10_PRU_s01 <= 0.2
 0 <= s_k10_PRU_s10 <= 0.2
 0 <= s_k10_PRU_s11 <= 0.2
 0 <= s_k10_PRU_s000 <= 0.2
 0 <= s_k10_PRU_s001 <= 0.2
 0 <= s_k10_PRU_s002 <= 0.2
 0 <= s_k10_PRU_s003 <= 0.2
 0 <= s_k10_PRU_s004 <= 0.2
 0 <= s_k10_PRU_s005 <= 0.2
 0 <= s_k10_PRU_s006 <= 0.2
 0 <= s_k10_PRU_s007 <= 0.2
 0 <= s_k10_PRU_s010 <= 0.2
 0 <= s_k10_PRU_s011 <= 0.2
 0 <= s_k10_PRU_s012 <= 0.2
 0 <= s_k10_PRU_s013 <= 0.2
 0 <= s_k10_PRU_s014 <= 0.2
 0 <= s_k10_PRU_s015 <= 0.2
 0 <= s_k10_PRU_s016 <= 0.2
 0 <= s_k10_PRU_s017 <= 0.2
 0 <= s_k10_PRU_s100 <= 0.2
 0 <= s_k10_PRU_s101 <= 0.2
 0 <= s_k10_PRU_s102 <= 0.2
 0 <= s_k10_PRU_s103 <= 0.2
 0 <= s_k10_PRU_s104 <= 0.2
 0 <= s_k10_PRU_s105 <= 0.2
 0 <= s_k10_PRU_s106 <= 0.2
 0 <= s_k10_PRU_s107 <= 0.2
 0 <= s_k10_PRU_s110 <= 0.2
 0 <= s_k10_PRU_s111 <= 0.2
 0 <= s_k10_PRU_s112 <= 0.2
 0 <= s_k10_PRU_s113 <= 0.2
 0 <= s_k10_PRU_s114 <= 0.2
 0 <= s_k10_PRU_s115 <= 0.2
 0 <= s_k10_PRU_s116 <= 0.2
 0 <= s_k10_PRU_s117 <= 0.2
End
