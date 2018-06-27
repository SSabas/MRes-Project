\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.09219 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.929246 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.19961 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.904474 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09762 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.979441 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.29638 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.967185 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.838448 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.13127 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.787284 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.18051 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.96277 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.820288 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.03069 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.827946 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.830418 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.16523 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.22002 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.0283 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.57163 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.07719 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.984353 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.22267 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.825204 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.987282 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.39757 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.938921 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.243 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.785214 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05896 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.48161 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10957 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.813378 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.4432 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.764696 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.0707 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.0261 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29009 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.916195 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.18906 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.746621 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.23044 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.947326 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.68562 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.13982 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.21377 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.973633 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.841978 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.06857 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.13433 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.713446 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.07804 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.832323 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.747014 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.28852 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.309 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.00032 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.6793 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.14658 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.901737 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.41958 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.669546 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.09027 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.69879 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.837843 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.09737 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.736741 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.06625 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.35049 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.923844 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.686162 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.67165 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.765732 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.37582 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.918854 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32272 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.870957 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.25381 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.793249 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.31116 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.937775 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.87806 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.08176 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.16893 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.01665 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.893784 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.36479 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.982179 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.622135 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.964103 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.862148 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.70613 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.875527 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.2957 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.915126 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 2.06235 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.3114 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.794732 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.46456 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.605844 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.893281 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.62741 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.01801 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.26052 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.619778 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.26558 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.72654 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.02546 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.586927 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.80931 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.600237 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.43829 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.941487 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18911 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.921304 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.11273 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.882207 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.0981 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.98855 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.54838 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.029 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.993845 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.19054 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.88003 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.21314 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.12224 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.731636 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.09097 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.788256 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.748505 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.878294 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.33565 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.949694 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.45661 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.07429 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.798863 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.26766 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.688664 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.02118 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.51374 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.873979 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.16587 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.766054 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.10997 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.46354 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.07333 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.741378 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.45094 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.685573 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.12541 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.908112 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20953 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.954994 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.18774 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.751309 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.54182 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.86253 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.27639 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.17572 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 2.00114 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.630717 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.16909 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.956458 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.897993 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.905315 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.640854 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.27752 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.765563 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.61763 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.925615 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.893531 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.4072 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.4383 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.26985 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.38506 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.99005 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.780941 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.841841 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.964548 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.807745 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.710312 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.1989 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.884077 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.771149 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.61887 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.28626 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.06539 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.73388 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.17645 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.22094 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.9967 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.10479 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.862981 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15941 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.921868 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.31466 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.957302 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.888731 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.08404 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.967123 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.03528 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.889438 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.767799 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.933105 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.763355 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.803572 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.883829 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.20045 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.958657 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.42562 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.00234 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.718842 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.97848 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.622989 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.855827 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.03134 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.933929 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.0767 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.685697 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.21947 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.33603 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.982958 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.746583 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.33227 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.743953 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.11711 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.981495 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.25214 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.910961 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.20279 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.846696 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.28551 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.944037 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.31913 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.990368 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.17402 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.76529 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.01539 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.994998 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.950045 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.798982 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.790786 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.879872 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.716638 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.01468 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.954511 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.947144 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.39431 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.16524 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.134 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.40428 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.928799 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.962929 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.10852 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.01968 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.18526 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.720636 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.23241 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.2537 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.855769 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.579172 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.21226 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.838038 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.4222 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.994083 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.30954 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.836318 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.17898 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.802783 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.66244 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.03597 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.42064 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.3011 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.71378 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.623174 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.912953 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.911892 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.24709 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.986931 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.617376 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.845558 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.594394 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.65433 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.933511 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.754266 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.28225 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.999095 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.35835 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.60217 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.02391 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.05371 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.04118 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.878821 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.941551 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.633694 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.0607 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.866803 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.714753 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.361481 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.4791 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.927409 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.75665 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.13787 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18004 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.930693 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13559 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.885476 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.17408 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.907763 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.26538 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.0346 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.962726 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.12081 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.848596 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.13829 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08703 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.718584 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.935517 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.888519 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.780619 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.971724 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.26942 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.04797 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.42809 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.08033 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.763825 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.20116 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.75427 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.935088 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.36145 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.899444 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.16182 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.714397 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.20112 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.53626 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.15944 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.791792 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.55779 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.799128 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.31532 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.03946 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13925 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.913339 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.05203 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.924335 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11589 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.0254 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.2642 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.00641 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.10262 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.0103 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.951295 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.15104 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.14345 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.881887 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.915501 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.876526 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.75365 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.949539 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.00449 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.937082 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.26534 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.06163 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.02702 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.21215 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.95159 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.04691 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.21905 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.06594 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11841 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.865358 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.05869 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.17477 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.987276 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.766861 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.22218 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.827402 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.21409 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.00553 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.236000927422411 z_k1_s0000
                                 - 0.353999357133829 z_k1_s0001
                                 - 0.31400075129936 z_k1_s0002
                                 - 0.201000062467468 z_k1_s0003
                                 - 0.185999924229612 z_k1_s0004
                                 - 0.227999172255729 z_k1_s0005
                                 - 0.378999248530292 z_k1_s0006
                                 - 0.135999938036706 z_k1_s0007
                                 - 0.370000958564199 z_k1_s0010
                                 - 0.280000796535258 z_k1_s0011
                                 - 0.175000497834536 z_k1_s0012
                                 - 0.728003255394803 z_k1_s0013
                                 - 0.276000803956232 z_k1_s0014
                                 - 0.532000987015598 z_k1_s0015
                                 - 0.478002074201365 z_k1_s0016
                                 - 0.451000972789899 z_k1_s0017
                                 - 0.225999853172243 z_k1_s0020
                                 - 0.129999555329489 z_k1_s0021
                                 - 0.102999853160873 z_k1_s0022
                                 - 0.113999844888726 z_k1_s0023
                                 - 0.260000382657837 z_k1_s0024
                                 - 0.156000483994474 z_k1_s0025
                                 - 0.0909999431304143 z_k1_s0026
                                 - 0.191999323686648 z_k1_s0027
                                 - 0.603996972631856 z_k1_s0030
                                 - 0.493997115100466 z_k1_s0031
                                 - 0.514000831007016 z_k1_s0032
                                 - 0.565998144391265 z_k1_s0033
                                 - 1.04599677434807 z_k1_s0034
                                 - 0.269999538050024 z_k1_s0035
                                 - 0.327999201003516 z_k1_s0036
                                 - 0.293999469547864 z_k1_s0037
                                 - 0.446998659551728 z_k1_s0040
                                 - 0.234999580189489 z_k1_s0041
                                 - 0.480998557593694 z_k1_s0042
                                 - 0.229999595183317 z_k1_s0043
                                 - 0.719997840888689 z_k1_s0044
                                 - 0.20499967015246 z_k1_s0045
                                 - 0.2119996491611 z_k1_s0046
                                 - 0.318999043393739 z_k1_s0047
                                 - 0.143000211449887 z_k1_s0050
                                 - 0.0750000631310403 z_k1_s0051
                                 - 0.135000486236022 z_k1_s0052
                                 - 0.172999491502004 z_k1_s0053
                                 - 0.262999815659352 z_k1_s0054
                                 - 0.243999692026112 z_k1_s0055
                                 - 0.0550000048960796 z_k1_s0056
                                 - 0.154000609869262 z_k1_s0057
                                 - 0.394000610255209 z_k1_s0060
                                 - 0.187999965888227 z_k1_s0061
                                 - 0.272000245455348 z_k1_s0062
                                 - 0.201999723816041 z_k1_s0063
                                 - 0.192000886439168 z_k1_s0064
                                 - 0.126000072950634 z_k1_s0065
                                 - 0.263000772216091 z_k1_s0066
                                 - 0.0950000398818249 z_k1_s0067
                                 - 0.619998813926467 z_k1_s0070
                                 - 1.0599958838139 z_k1_s0071
                                 - 0.507998835411099 z_k1_s0072
                                 - 0.250999471638068 z_k1_s0073
                                 - 0.441999025928477 z_k1_s0074
                                 - 0.387999000715781 z_k1_s0075
                                 - 1.0959962326217 z_k1_s0076
                                 - 0.614998903813998 z_k1_s0077
                                 - 0.371001482360285 z_k1_s0100
                                 - 0.114000397306184 z_k1_s0101
                                 - 0.374001236246359 z_k1_s0102
                                 - 0.141000445589061 z_k1_s0103
                                 - 0.235001032816155 z_k1_s0104
                                 - 0.296000671183169 z_k1_s0105
                                 - 0.127000491483382 z_k1_s0106
                                 - 0.0830003870074884 z_k1_s0107
                                 - 0.295998518002833 z_k1_s0110
                                 - 0.22499938076419 z_k1_s0111
                                 - 0.097999726408414 z_k1_s0112
                                 - 0.135999559746547 z_k1_s0113
                                 - 0.1479996082004 z_k1_s0114
                                 - 0.395998630785788 z_k1_s0115
                                 - 0.105999700511152 z_k1_s0116
                                 - 0.340999790951604 z_k1_s0117
                                 - 0.41599865148826 z_k1_s0120
                                 - 0.353999681938854 z_k1_s0121
                                 - 0.263999845174671 z_k1_s0122
                                 - 0.260000068429421 z_k1_s0123
                                 - 0.43799985358516 z_k1_s0124
                                 - 0.346998857635654 z_k1_s0125
                                 - 0.233999818586677 z_k1_s0126
                                 - 0.116999787793437 z_k1_s0127
                                 - 0.0689999895075108 z_k1_s0130
                                 - 0.123999637932405 z_k1_s0131
                                 - 0.121999945320533 z_k1_s0132
                                 - 0.0470000210775744 z_k1_s0133
                                 - 0.107000303231875 z_k1_s0134
                                 - 0.159000101139022 z_k1_s0135
                                 - 0.111999924261478 z_k1_s0136
                                 - 0.0390000042303302 z_k1_s0137
                                 - 0.295000406991766 z_k1_s0140
                                 - 0.208000519465997 z_k1_s0141
                                 - 0.205000577923873 z_k1_s0142
                                 - 0.188000469517343 z_k1_s0143
                                 - 0.407002269508417 z_k1_s0144
                                 - 0.135000271203237 z_k1_s0145
                                 - 0.5340012017287 z_k1_s0146
                                 - 0.666001795190512 z_k1_s0147
                                 - 0.240999262489895 z_k1_s0150
                                 - 0.250000016092002 z_k1_s0151
                                 - 0.454999816167785 z_k1_s0152
                                 - 0.208999345677982 z_k1_s0153
                                 - 0.0919998496341068 z_k1_s0154
                                 - 0.193999273672575 z_k1_s0155
                                 - 0.151999782456301 z_k1_s0156
                                 - 0.183000523266527 z_k1_s0157
                                 - 0.0620000595634228 z_k1_s0160
                                 - 0.0750000214637953 z_k1_s0161
                                 - 0.169000271405185 z_k1_s0162
                                 - 0.0470000552706637 z_k1_s0163
                                 - 0.0600000171710362 z_k1_s0164
                                 - 0.0510000006553746 z_k1_s0165
                                 - 0.112999781418673 z_k1_s0166
                                 - 0.119999755541948 z_k1_s0167
                                 - 0.260999539688232 z_k1_s0170
                                 - 0.407999814914302 z_k1_s0171
                                 - 0.10599987897292 z_k1_s0172
                                 - 0.361999744605445 z_k1_s0173
                                 - 0.174999751936482 z_k1_s0174
                                 - 0.316999423107951 z_k1_s0175
                                 - 0.444999820923661 z_k1_s0176
                                 - 0.250999726577371 z_k1_s0177
                                 - 0.354001542705104 z_k1_s1000
                                 - 0.187000332635821 z_k1_s1001
                                 - 0.166000396379079 z_k1_s1002
                                 - 0.412000011630285 z_k1_s1003
                                 - 0.268999619684944 z_k1_s1004
                                 - 0.232999561901591 z_k1_s1005
                                 - 0.153000343029463 z_k1_s1006
                                 - 0.175000313374704 z_k1_s1007
                                 - 0.689998544987068 z_k1_s1010
                                 - 0.370998294733973 z_k1_s1011
                                 - 0.272999770420841 z_k1_s1012
                                 - 0.551001376164083 z_k1_s1013
                                 - 0.451999508327587 z_k1_s1014
                                 - 0.549000918915018 z_k1_s1015
                                 - 0.244000062306805 z_k1_s1016
                                 - 0.330999878848652 z_k1_s1017
                                 - 0.0679999677611264 z_k1_s1020
                                 - 0.227999778610886 z_k1_s1021
                                 - 0.209000439059693 z_k1_s1022
                                 - 0.0639999243399038 z_k1_s1023
                                 - 0.172999663068861 z_k1_s1024
                                 - 0.103999973352301 z_k1_s1025
                                 - 0.195999944840877 z_k1_s1026
                                 - 0.242000508384908 z_k1_s1027
                                 - 0.542000199098867 z_k1_s1030
                                 - 0.368000045151858 z_k1_s1031
                                 - 0.372001323405505 z_k1_s1032
                                 - 0.226999346780109 z_k1_s1033
                                 - 0.163000187555025 z_k1_s1034
                                 - 0.155000292651889 z_k1_s1035
                                 - 0.282001396743498 z_k1_s1036
                                 - 0.109000231557821 z_k1_s1037
                                 - 0.0459999593298621 z_k1_s1040
                                 - 0.119000402187977 z_k1_s1041
                                 - 0.127000420914902 z_k1_s1042
                                 - 0.0729999483582557 z_k1_s1043
                                 - 0.195999964309808 z_k1_s1044
                                 - 0.132999693210307 z_k1_s1045
                                 - 0.0889999858121061 z_k1_s1046
                                 - 0.205999740468535 z_k1_s1047
                                 - 0.320999460588268 z_k1_s1050
                                 - 0.133999435227243 z_k1_s1051
                                 - 0.169999339601666 z_k1_s1052
                                 - 0.200999779340192 z_k1_s1053
                                 - 0.148000442205065 z_k1_s1054
                                 - 0.0769998998840703 z_k1_s1055
                                 - 0.0759998816570254 z_k1_s1056
                                 - 0.126000291809808 z_k1_s1057
                                 - 0.207999941122178 z_k1_s1060
                                 - 0.278000324903321 z_k1_s1061
                                 - 0.113999928672767 z_k1_s1062
                                 - 0.169999964897922 z_k1_s1063
                                 - 0.115999920295102 z_k1_s1064
                                 - 0.233999426012901 z_k1_s1065
                                 - 0.0909999573159678 z_k1_s1066
                                 - 0.142999874896572 z_k1_s1067
                                 - 0.116999801150214 z_k1_s1070
                                 - 0.236000399720767 z_k1_s1071
                                 - 0.467999204600856 z_k1_s1072
                                 - 0.390998759878808 z_k1_s1073
                                 - 0.246999905538921 z_k1_s1074
                                 - 0.134999755542574 z_k1_s1075
                                 - 0.152999709934934 z_k1_s1076
                                 - 0.204998775691675 z_k1_s1077
                                 - 0.222000104087018 z_k1_s1100
                                 - 0.729002067076016 z_k1_s1101
                                 - 0.395999734744128 z_k1_s1102
                                 - 0.51199852131493 z_k1_s1103
                                 - 0.223000068612613 z_k1_s1104
                                 - 0.61899871525606 z_k1_s1105
                                 - 0.62599882963545 z_k1_s1106
                                 - 0.300000238685221 z_k1_s1107
                                 - 0.718001462453807 z_k1_s1110
                                 - 0.628998861787074 z_k1_s1111
                                 - 0.487001683196514 z_k1_s1112
                                 - 0.842999677861122 z_k1_s1113
                                 - 0.196000113503214 z_k1_s1114
                                 - 0.601999048731057 z_k1_s1115
                                 - 0.424998895585798 z_k1_s1116
                                 - 0.236000204193679 z_k1_s1117
                                 - 1.27699815690187 z_k1_s1120
                                 - 1.18300310306413 z_k1_s1121
                                 - 0.767999582875544 z_k1_s1122
                                 - 0.694999392152314 z_k1_s1123
                                 - 0.993998143383701 z_k1_s1124
                                 - 1.13100316405012 z_k1_s1125
                                 - 2.4059985599157 z_k1_s1126
                                 - 0.526999174676716 z_k1_s1127
                                 - 0.758001265357401 z_k1_s1130
                                 - 0.649999673854674 z_k1_s1131
                                 - 0.905001404776389 z_k1_s1132
                                 - 0.389999536552987 z_k1_s1133
                                 - 0.323000005795669 z_k1_s1134
                                 - 1.58499754150677 z_k1_s1135
                                 - 1.5279981705071 z_k1_s1136
                                 - 0.55499982965584 z_k1_s1137
                                 - 0.297000243105172 z_k1_s1140
                                 - 0.40400010799891 z_k1_s1141
                                 - 0.224000246977651 z_k1_s1142
                                 - 0.403000220599925 z_k1_s1143
                                 - 0.575000982429307 z_k1_s1144
                                 - 0.982001163425253 z_k1_s1145
                                 - 0.554000882250137 z_k1_s1146
                                 - 0.669000257135834 z_k1_s1147
                                 - 0.504999561804044 z_k1_s1150
                                 - 0.401999239821579 z_k1_s1151
                                 - 0.830997796303645 z_k1_s1152
                                 - 1.73199800498085 z_k1_s1153
                                 - 0.605999097605346 z_k1_s1154
                                 - 0.390999309019433 z_k1_s1155
                                 - 0.69799697838895 z_k1_s1156
                                 - 1.1110011698061 z_k1_s1157
                                 - 0.285999958126021 z_k1_s1160
                                 - 0.293999916318362 z_k1_s1161
                                 - 0.6909993502381 z_k1_s1162
                                 - 0.501002019419847 z_k1_s1163
                                 - 0.794001549839206 z_k1_s1164
                                 - 0.427999774790007 z_k1_s1165
                                 - 0.279999877731777 z_k1_s1166
                                 - 1.19600201925405 z_k1_s1167
                                 - 0.585001296810384 z_k1_s1170
                                 - 1.18999811653189 z_k1_s1171
                                 - 2.2980002038599 z_k1_s1172
                                 - 2.33000742578598 z_k1_s1173
                                 - 0.540000929339828 z_k1_s1174
                                 - 1.55700807407211 z_k1_s1175
                                 - 1.16300253645868 z_k1_s1176
                                 - 1.93800886511826 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.01473 w_k1_ADM_s000
                                 + 0.587239 w_k1_AXP_s000
                                 + 0.724538 w_k1_DIS_s000
                                 + 0.75351 w_k1_FDX_s000
                                 + 0.861829 w_k1_IBM_s000
                                 + 0.73266 w_k1_INTC_s000
                                 + 0.848362 w_k1_KO_s000
                                 + 0.668824 w_k1_MAT_s000
                                 + 0.905925 w_k1_MSFT_s000
                                 + 0.877107 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.38626 w_k1_ADM_s000
                                 + 1.23265 w_k1_AXP_s000
                                 + 1.07622 w_k1_DIS_s000
                                 + 1.15524 w_k1_FDX_s000
                                 + 0.944422 w_k1_IBM_s000
                                 + 1.36741 w_k1_INTC_s000
                                 + 1.04791 w_k1_KO_s000 + 1.10837 w_k1_MAT_s000
                                 + 1.30303 w_k1_MSFT_s000
                                 + 1.07764 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.884504 w_k1_ADM_s000
                                 + 0.948998 w_k1_AXP_s000
                                 + 0.869168 w_k1_DIS_s000
                                 + 1.03023 w_k1_FDX_s000
                                 + 0.742606 w_k1_IBM_s000
                                 + 1.00634 w_k1_INTC_s000
                                 + 0.895787 w_k1_KO_s000
                                 + 1.01352 w_k1_MAT_s000
                                 + 1.07666 w_k1_MSFT_s000 + 1.0384 w_k1_PG_s000
                                 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.39207 w_k1_ADM_s000
                                 + 1.28802 w_k1_AXP_s000
                                 + 1.75141 w_k1_DIS_s000 + 1.3755 w_k1_FDX_s000
                                 + 0.920622 w_k1_IBM_s000
                                 + 1.37196 w_k1_INTC_s000
                                 + 1.34887 w_k1_KO_s000
                                 + 0.998835 w_k1_MAT_s000
                                 + 1.59592 w_k1_MSFT_s000
                                 + 1.27112 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.13925 w_k1_ADM_s000
                                 + 0.655062 w_k1_AXP_s000
                                 + 0.932429 w_k1_DIS_s000
                                 + 0.995722 w_k1_FDX_s000
                                 + 0.758266 w_k1_IBM_s000
                                 + 0.921645 w_k1_INTC_s000
                                 + 1.01082 w_k1_KO_s000 + 0.66566 w_k1_MAT_s000
                                 + 1.14777 w_k1_MSFT_s000
                                 + 1.06122 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.17297 w_k1_ADM_s000
                                 + 1.16195 w_k1_AXP_s000
                                 + 1.32237 w_k1_DIS_s000
                                 + 1.11044 w_k1_FDX_s000 + 1.6446 w_k1_IBM_s000
                                 + 1.02071 w_k1_INTC_s000
                                 + 1.41692 w_k1_KO_s000 + 1.56574 w_k1_MAT_s000
                                 + 1.26709 w_k1_MSFT_s000
                                 + 1.13402 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.75689 w_k1_ADM_s000
                                 + 0.874208 w_k1_AXP_s000
                                 + 0.769491 w_k1_DIS_s000
                                 + 0.839392 w_k1_FDX_s000
                                 + 1.20177 w_k1_IBM_s000
                                 + 0.912661 w_k1_INTC_s000
                                 + 0.898616 w_k1_KO_s000 + 1.0044 w_k1_MAT_s000
                                 + 0.838229 w_k1_MSFT_s000
                                 + 1.00226 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.910586 w_k1_ADM_s000
                                 + 0.596773 w_k1_AXP_s000
                                 + 0.502719 w_k1_DIS_s000
                                 + 0.753395 w_k1_FDX_s000
                                 + 0.647855 w_k1_IBM_s000
                                 + 0.698621 w_k1_INTC_s000
                                 + 0.708793 w_k1_KO_s000
                                 + 0.692706 w_k1_MAT_s000
                                 + 0.763547 w_k1_MSFT_s000
                                 + 0.84344 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.15893 w_k1_ADM_s001
                                 + 0.960499 w_k1_AXP_s001
                                 + 0.773972 w_k1_DIS_s001
                                 + 0.978371 w_k1_FDX_s001
                                 + 0.708916 w_k1_IBM_s001
                                 + 0.915391 w_k1_INTC_s001
                                 + 0.80631 w_k1_KO_s001
                                 + 0.672666 w_k1_MAT_s001
                                 + 0.904233 w_k1_MSFT_s001
                                 + 1.02009 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.781399 w_k1_ADM_s001
                                 + 0.656776 w_k1_AXP_s001
                                 + 0.56088 w_k1_DIS_s001
                                 + 0.796558 w_k1_FDX_s001
                                 + 0.594951 w_k1_IBM_s001
                                 + 0.569887 w_k1_INTC_s001
                                 + 0.705667 w_k1_KO_s001
                                 + 0.619122 w_k1_MAT_s001
                                 + 0.695952 w_k1_MSFT_s001
                                 + 0.877647 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.49189 w_k1_ADM_s001
                                 + 1.12676 w_k1_AXP_s001
                                 + 1.68844 w_k1_DIS_s001
                                 + 1.33428 w_k1_FDX_s001 + 1.2442 w_k1_IBM_s001
                                 + 1.12866 w_k1_INTC_s001
                                 + 1.37375 w_k1_KO_s001 + 1.10359 w_k1_MAT_s001
                                 + 1.3231 w_k1_MSFT_s001 + 1.2917 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.992583 w_k1_ADM_s001
                                 + 0.910009 w_k1_AXP_s001
                                 + 0.990272 w_k1_DIS_s001
                                 + 0.923666 w_k1_FDX_s001
                                 + 1.13512 w_k1_IBM_s001
                                 + 0.993761 w_k1_INTC_s001
                                 + 1.08827 w_k1_KO_s001
                                 + 0.977319 w_k1_MAT_s001
                                 + 0.934124 w_k1_MSFT_s001
                                 + 1.11153 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.18163 w_k1_ADM_s001
                                 + 1.33936 w_k1_AXP_s001
                                 + 1.14771 w_k1_DIS_s001
                                 + 1.21925 w_k1_FDX_s001
                                 + 1.12735 w_k1_IBM_s001
                                 + 1.01346 w_k1_INTC_s001
                                 + 1.17733 w_k1_KO_s001 + 1.54491 w_k1_MAT_s001
                                 + 1.10807 w_k1_MSFT_s001
                                 + 1.29731 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.887704 w_k1_ADM_s001
                                 + 0.879118 w_k1_AXP_s001
                                 + 0.680865 w_k1_DIS_s001
                                 + 0.831999 w_k1_FDX_s001
                                 + 0.889881 w_k1_IBM_s001
                                 + 0.88861 w_k1_INTC_s001 + 0.8119 w_k1_KO_s001
                                 + 0.839157 w_k1_MAT_s001
                                 + 0.67669 w_k1_MSFT_s001
                                 + 0.870834 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.24985 w_k1_ADM_s001
                                 + 1.34496 w_k1_AXP_s001
                                 + 1.23251 w_k1_DIS_s001
                                 + 1.10142 w_k1_FDX_s001
                                 + 1.53162 w_k1_IBM_s001
                                 + 1.20436 w_k1_INTC_s001
                                 + 1.52078 w_k1_KO_s001 + 1.50614 w_k1_MAT_s001
                                 + 1.01185 w_k1_MSFT_s001
                                 + 1.30573 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.13808 w_k1_ADM_s001
                                 + 0.900152 w_k1_AXP_s001
                                 + 1.10113 w_k1_DIS_s001
                                 + 1.15357 w_k1_FDX_s001
                                 + 0.861786 w_k1_IBM_s001
                                 + 1.09111 w_k1_INTC_s001
                                 + 1.04757 w_k1_KO_s001
                                 + 0.500853 w_k1_MAT_s001
                                 + 1.05706 w_k1_MSFT_s001
                                 + 1.01607 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.49745 w_k1_ADM_s002
                                 + 1.16822 w_k1_AXP_s002
                                 + 0.929698 w_k1_DIS_s002
                                 + 0.965486 w_k1_FDX_s002
                                 + 1.32732 w_k1_IBM_s002
                                 + 1.07002 w_k1_INTC_s002
                                 + 1.17055 w_k1_KO_s002 + 1.34011 w_k1_MAT_s002
                                 + 1.02069 w_k1_MSFT_s002
                                 + 1.09711 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.982892 w_k1_ADM_s002
                                 + 0.614186 w_k1_AXP_s002
                                 + 0.43572 w_k1_DIS_s002
                                 + 0.584269 w_k1_FDX_s002
                                 + 0.736853 w_k1_IBM_s002
                                 + 0.787024 w_k1_INTC_s002
                                 + 0.771888 w_k1_KO_s002
                                 + 0.912169 w_k1_MAT_s002
                                 + 0.696305 w_k1_MSFT_s002
                                 + 0.841669 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.17428 w_k1_ADM_s002
                                 + 0.959041 w_k1_AXP_s002
                                 + 1.05543 w_k1_DIS_s002
                                 + 1.07722 w_k1_FDX_s002
                                 + 0.830445 w_k1_IBM_s002
                                 + 1.33625 w_k1_INTC_s002
                                 + 1.07091 w_k1_KO_s002
                                 + 0.712957 w_k1_MAT_s002
                                 + 1.14149 w_k1_MSFT_s002
                                 + 0.903288 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.81446 w_k1_ADM_s002
                                 + 1.78055 w_k1_AXP_s002
                                 + 2.13433 w_k1_DIS_s002
                                 + 1.45462 w_k1_FDX_s002
                                 + 1.46814 w_k1_IBM_s002
                                 + 1.51131 w_k1_INTC_s002
                                 + 1.70349 w_k1_KO_s002 + 2.27095 w_k1_MAT_s002
                                 + 1.52975 w_k1_MSFT_s002
                                 + 1.60716 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.22314 w_k1_ADM_s002
                                 + 0.90889 w_k1_AXP_s002
                                 + 1.13863 w_k1_DIS_s002
                                 + 1.04392 w_k1_FDX_s002
                                 + 0.999241 w_k1_IBM_s002
                                 + 1.22818 w_k1_INTC_s002
                                 + 1.10689 w_k1_KO_s002 + 1.12237 w_k1_MAT_s002
                                 + 1.14048 w_k1_MSFT_s002
                                 + 1.13916 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.975051 w_k1_ADM_s002
                                 + 0.780926 w_k1_AXP_s002
                                 + 0.600289 w_k1_DIS_s002
                                 + 0.782489 w_k1_FDX_s002
                                 + 1.01571 w_k1_IBM_s002
                                 + 1.01445 w_k1_INTC_s002
                                 + 0.897009 w_k1_KO_s002 + 1.1173 w_k1_MAT_s002
                                 + 0.885723 w_k1_MSFT_s002
                                 + 1.08843 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.51743 w_k1_ADM_s002
                                 + 1.45149 w_k1_AXP_s002 + 1.4754 w_k1_DIS_s002
                                 + 1.35689 w_k1_FDX_s002
                                 + 0.933793 w_k1_IBM_s002
                                 + 1.21087 w_k1_INTC_s002
                                 + 1.22461 w_k1_KO_s002 + 1.50464 w_k1_MAT_s002
                                 + 1.24481 w_k1_MSFT_s002
                                 + 1.22135 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.00256 w_k1_ADM_s002
                                 + 0.782407 w_k1_AXP_s002
                                 + 0.780475 w_k1_DIS_s002
                                 + 0.884884 w_k1_FDX_s002
                                 + 0.751591 w_k1_IBM_s002
                                 + 1.08892 w_k1_INTC_s002
                                 + 0.912892 w_k1_KO_s002
                                 + 0.667235 w_k1_MAT_s002
                                 + 0.88422 w_k1_MSFT_s002
                                 + 0.889864 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.2546 w_k1_ADM_s003
                                 + 1.35967 w_k1_AXP_s003
                                 + 1.53741 w_k1_DIS_s003 + 1.3209 w_k1_FDX_s003
                                 + 1.03629 w_k1_IBM_s003
                                 + 1.28509 w_k1_INTC_s003
                                 + 1.23243 w_k1_KO_s003 + 1.00402 w_k1_MAT_s003
                                 + 1.26181 w_k1_MSFT_s003
                                 + 1.14254 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.11331 w_k1_ADM_s003
                                 + 1.14948 w_k1_AXP_s003
                                 + 1.08928 w_k1_DIS_s003
                                 + 1.09052 w_k1_FDX_s003
                                 + 1.14023 w_k1_IBM_s003
                                 + 0.952283 w_k1_INTC_s003
                                 + 1.05065 w_k1_KO_s003 + 1.40544 w_k1_MAT_s003
                                 + 1.05662 w_k1_MSFT_s003
                                 + 1.10435 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.871255 w_k1_ADM_s003
                                 + 0.890621 w_k1_AXP_s003
                                 + 0.842413 w_k1_DIS_s003
                                 + 0.939819 w_k1_FDX_s003
                                 + 0.736251 w_k1_IBM_s003
                                 + 0.996966 w_k1_INTC_s003
                                 + 0.928774 w_k1_KO_s003
                                 + 0.685758 w_k1_MAT_s003
                                 + 0.936292 w_k1_MSFT_s003
                                 + 0.962743 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.808366 w_k1_ADM_s003
                                 + 0.704302 w_k1_AXP_s003
                                 + 0.67011 w_k1_DIS_s003
                                 + 0.708959 w_k1_FDX_s003
                                 + 0.991722 w_k1_IBM_s003
                                 + 0.689427 w_k1_INTC_s003
                                 + 0.820324 w_k1_KO_s003
                                 + 0.75845 w_k1_MAT_s003
                                 + 0.73955 w_k1_MSFT_s003
                                 + 0.856198 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.961636 w_k1_ADM_s003
                                 + 0.885611 w_k1_AXP_s003
                                 + 0.985599 w_k1_DIS_s003
                                 + 0.915128 w_k1_FDX_s003
                                 + 1.31623 w_k1_IBM_s003
                                 + 0.912828 w_k1_INTC_s003
                                 + 1.33982 w_k1_KO_s003 + 1.34621 w_k1_MAT_s003
                                 + 0.879866 w_k1_MSFT_s003
                                 + 1.0643 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.20911 w_k1_ADM_s003
                                 + 1.47852 w_k1_AXP_s003
                                 + 1.57126 w_k1_DIS_s003
                                 + 1.15259 w_k1_FDX_s003
                                 + 1.69259 w_k1_IBM_s003
                                 + 1.04109 w_k1_INTC_s003
                                 + 1.55712 w_k1_KO_s003 + 2.03303 w_k1_MAT_s003
                                 + 1.14172 w_k1_MSFT_s003
                                 + 1.38433 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.810187 w_k1_ADM_s003
                                 + 0.65091 w_k1_AXP_s003
                                 + 0.56011 w_k1_DIS_s003
                                 + 0.755974 w_k1_FDX_s003
                                 + 0.574986 w_k1_IBM_s003
                                 + 0.699606 w_k1_INTC_s003
                                 + 0.637377 w_k1_KO_s003
                                 + 0.68103 w_k1_MAT_s003
                                 + 0.725869 w_k1_MSFT_s003
                                 + 0.76892 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.931183 w_k1_ADM_s003
                                 + 1.15491 w_k1_AXP_s003
                                 + 0.898257 w_k1_DIS_s003
                                 + 1.03952 w_k1_FDX_s003
                                 + 0.86616 w_k1_IBM_s003
                                 + 1.00704 w_k1_INTC_s003
                                 + 0.86666 w_k1_KO_s003 + 1.3 w_k1_MAT_s003
                                 + 0.927208 w_k1_MSFT_s003
                                 + 0.983868 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.07414 w_k1_ADM_s004
                                 + 0.903699 w_k1_AXP_s004
                                 + 0.8313 w_k1_DIS_s004 + 0.86266 w_k1_FDX_s004
                                 + 1.17269 w_k1_IBM_s004
                                 + 0.765057 w_k1_INTC_s004
                                 + 0.879928 w_k1_KO_s004
                                 + 1.20012 w_k1_MAT_s004
                                 + 0.794611 w_k1_MSFT_s004
                                 + 0.974511 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.11472 w_k1_ADM_s004
                                 + 1.10045 w_k1_AXP_s004
                                 + 1.00548 w_k1_DIS_s004
                                 + 0.980673 w_k1_FDX_s004
                                 + 1.26989 w_k1_IBM_s004
                                 + 0.824745 w_k1_INTC_s004
                                 + 1.19054 w_k1_KO_s004 + 1.34009 w_k1_MAT_s004
                                 + 1.00317 w_k1_MSFT_s004
                                 + 1.03802 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.00767 w_k1_ADM_s004
                                 + 1.04366 w_k1_AXP_s004
                                 + 1.31353 w_k1_DIS_s004
                                 + 1.11726 w_k1_FDX_s004
                                 + 0.96176 w_k1_IBM_s004
                                 + 1.11548 w_k1_INTC_s004
                                 + 1.05055 w_k1_KO_s004 + 1.18403 w_k1_MAT_s004
                                 + 1.31652 w_k1_MSFT_s004
                                 + 1.12758 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.2761 w_k1_ADM_s004
                                 + 1.07468 w_k1_AXP_s004
                                 + 1.13142 w_k1_DIS_s004
                                 + 1.16006 w_k1_FDX_s004
                                 + 0.716696 w_k1_IBM_s004
                                 + 1.05612 w_k1_INTC_s004
                                 + 0.992115 w_k1_KO_s004
                                 + 0.877507 w_k1_MAT_s004
                                 + 1.19103 w_k1_MSFT_s004
                                 + 0.971418 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.966109 w_k1_ADM_s004
                                 + 0.80174 w_k1_AXP_s004
                                 + 0.884683 w_k1_DIS_s004
                                 + 0.853558 w_k1_FDX_s004
                                 + 0.772798 w_k1_IBM_s004
                                 + 0.735975 w_k1_INTC_s004
                                 + 0.863911 w_k1_KO_s004
                                 + 1.04181 w_k1_MAT_s004
                                 + 0.969021 w_k1_MSFT_s004
                                 + 0.960046 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.43424 w_k1_ADM_s004
                                 + 1.23142 w_k1_AXP_s004
                                 + 1.31323 w_k1_DIS_s004
                                 + 1.10773 w_k1_FDX_s004
                                 + 1.75878 w_k1_IBM_s004
                                 + 0.86802 w_k1_INTC_s004
                                 + 1.36191 w_k1_KO_s004 + 2.1478 w_k1_MAT_s004
                                 + 1.29563 w_k1_MSFT_s004
                                 + 1.23167 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.909633 w_k1_ADM_s004
                                 + 0.808204 w_k1_AXP_s004
                                 + 0.585761 w_k1_DIS_s004
                                 + 0.724065 w_k1_FDX_s004
                                 + 0.688651 w_k1_IBM_s004
                                 + 0.821319 w_k1_INTC_s004
                                 + 0.703271 w_k1_KO_s004
                                 + 0.71043 w_k1_MAT_s004
                                 + 0.843627 w_k1_MSFT_s004
                                 + 0.895232 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.47201 w_k1_ADM_s004
                                 + 1.30732 w_k1_AXP_s004
                                 + 1.44919 w_k1_DIS_s004
                                 + 1.18913 w_k1_FDX_s004
                                 + 1.25921 w_k1_IBM_s004
                                 + 0.831001 w_k1_INTC_s004
                                 + 1.19759 w_k1_KO_s004 + 1.44264 w_k1_MAT_s004
                                 + 1.11301 w_k1_MSFT_s004 + 1.1743 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.838523 w_k1_ADM_s005
                                 + 0.991547 w_k1_AXP_s005
                                 + 0.982943 w_k1_DIS_s005
                                 + 1.00607 w_k1_FDX_s005 + 1.1552 w_k1_IBM_s005
                                 + 1.09513 w_k1_INTC_s005
                                 + 1.13712 w_k1_KO_s005 + 1.15936 w_k1_MAT_s005
                                 + 1.03042 w_k1_MSFT_s005
                                 + 1.00877 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.778219 w_k1_ADM_s005
                                 + 0.841212 w_k1_AXP_s005
                                 + 0.766593 w_k1_DIS_s005
                                 + 0.975354 w_k1_FDX_s005
                                 + 0.811114 w_k1_IBM_s005
                                 + 1.1931 w_k1_INTC_s005
                                 + 0.954908 w_k1_KO_s005
                                 + 0.601895 w_k1_MAT_s005
                                 + 1.02383 w_k1_MSFT_s005
                                 + 0.924148 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.897901 w_k1_ADM_s005
                                 + 0.687076 w_k1_AXP_s005
                                 + 0.68997 w_k1_DIS_s005
                                 + 0.832626 w_k1_FDX_s005
                                 + 0.744792 w_k1_IBM_s005
                                 + 0.957083 w_k1_INTC_s005
                                 + 0.77046 w_k1_KO_s005
                                 + 0.381103 w_k1_MAT_s005
                                 + 0.865292 w_k1_MSFT_s005
                                 + 0.786241 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.21582 w_k1_ADM_s005
                                 + 1.22029 w_k1_AXP_s005
                                 + 1.46597 w_k1_DIS_s005
                                 + 1.08609 w_k1_FDX_s005
                                 + 1.56421 w_k1_IBM_s005
                                 + 1.30091 w_k1_INTC_s005
                                 + 1.68752 w_k1_KO_s005 + 1.24229 w_k1_MAT_s005
                                 + 1.22167 w_k1_MSFT_s005
                                 + 1.09688 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.886142 w_k1_ADM_s005
                                 + 1.0933 w_k1_AXP_s005 + 1.4414 w_k1_DIS_s005
                                 + 1.17073 w_k1_FDX_s005
                                 + 1.13832 w_k1_IBM_s005
                                 + 1.01924 w_k1_INTC_s005
                                 + 1.21796 w_k1_KO_s005 + 1.15517 w_k1_MAT_s005
                                 + 1.18378 w_k1_MSFT_s005
                                 + 1.11225 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.996337 w_k1_ADM_s005
                                 + 0.749787 w_k1_AXP_s005
                                 + 0.846215 w_k1_DIS_s005
                                 + 0.84567 w_k1_FDX_s005
                                 + 1.02209 w_k1_IBM_s005
                                 + 0.851095 w_k1_INTC_s005
                                 + 1.12375 w_k1_KO_s005
                                 + 0.774353 w_k1_MAT_s005
                                 + 0.915283 w_k1_MSFT_s005
                                 + 0.941047 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.07367 w_k1_ADM_s005
                                 + 1.04391 w_k1_AXP_s005
                                 + 1.11231 w_k1_DIS_s005
                                 + 1.20925 w_k1_FDX_s005
                                 + 0.786782 w_k1_IBM_s005
                                 + 1.06672 w_k1_INTC_s005 + 1.0672 w_k1_KO_s005
                                 + 1.10792 w_k1_MAT_s005
                                 + 1.2926 w_k1_MSFT_s005 + 1.25896 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.851341 w_k1_ADM_s005
                                 + 0.816397 w_k1_AXP_s005
                                 + 1.10438 w_k1_DIS_s005
                                 + 0.989869 w_k1_FDX_s005
                                 + 0.821221 w_k1_IBM_s005
                                 + 1.06474 w_k1_INTC_s005
                                 + 0.979973 w_k1_KO_s005
                                 + 0.802027 w_k1_MAT_s005
                                 + 1.10761 w_k1_MSFT_s005
                                 + 0.912907 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.783259 w_k1_ADM_s006
                                 + 0.744948 w_k1_AXP_s006
                                 + 0.72584 w_k1_DIS_s006
                                 + 0.741655 w_k1_FDX_s006
                                 + 0.738887 w_k1_IBM_s006
                                 + 1.06616 w_k1_INTC_s006
                                 + 0.704542 w_k1_KO_s006
                                 + 0.640211 w_k1_MAT_s006
                                 + 0.809987 w_k1_MSFT_s006
                                 + 0.734843 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.950843 w_k1_ADM_s006
                                 + 1.01962 w_k1_AXP_s006
                                 + 1.00387 w_k1_DIS_s006
                                 + 0.907425 w_k1_FDX_s006
                                 + 1.10264 w_k1_IBM_s006
                                 + 1.0993 w_k1_INTC_s006
                                 + 0.984931 w_k1_KO_s006
                                 + 0.993186 w_k1_MAT_s006
                                 + 0.871459 w_k1_MSFT_s006
                                 + 0.918205 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.30627 w_k1_ADM_s006
                                 + 1.5698 w_k1_AXP_s006 + 1.67568 w_k1_DIS_s006
                                 + 1.25003 w_k1_FDX_s006
                                 + 1.44687 w_k1_IBM_s006
                                 + 1.31915 w_k1_INTC_s006
                                 + 1.39706 w_k1_KO_s006 + 1.13022 w_k1_MAT_s006
                                 + 1.1976 w_k1_MSFT_s006 + 1.15905 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.15811 w_k1_ADM_s006
                                 + 0.848066 w_k1_AXP_s006
                                 + 1.23185 w_k1_DIS_s006
                                 + 1.07395 w_k1_FDX_s006
                                 + 0.987595 w_k1_IBM_s006
                                 + 1.01093 w_k1_INTC_s006
                                 + 1.05669 w_k1_KO_s006
                                 + 0.707124 w_k1_MAT_s006
                                 + 0.96691 w_k1_MSFT_s006
                                 + 1.03173 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.920116 w_k1_ADM_s006
                                 + 1.46923 w_k1_AXP_s006
                                 + 1.04917 w_k1_DIS_s006
                                 + 1.15657 w_k1_FDX_s006
                                 + 0.898449 w_k1_IBM_s006
                                 + 1.31612 w_k1_INTC_s006
                                 + 0.834098 w_k1_KO_s006 + 1.0278 w_k1_MAT_s006
                                 + 1.00326 w_k1_MSFT_s006
                                 + 0.962291 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.779204 w_k1_ADM_s006
                                 + 0.704605 w_k1_AXP_s006
                                 + 0.884418 w_k1_DIS_s006
                                 + 0.827718 w_k1_FDX_s006
                                 + 1.11928 w_k1_IBM_s006
                                 + 0.942687 w_k1_INTC_s006 + 1.007 w_k1_KO_s006
                                 + 0.816134 w_k1_MAT_s006
                                 + 0.863848 w_k1_MSFT_s006
                                 + 0.893175 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.992545 w_k1_ADM_s006
                                 + 1.05399 w_k1_AXP_s006
                                 + 0.98266 w_k1_DIS_s006
                                 + 1.00422 w_k1_FDX_s006
                                 + 0.942904 w_k1_IBM_s006
                                 + 1.00486 w_k1_INTC_s006
                                 + 0.925221 w_k1_KO_s006
                                 + 0.810149 w_k1_MAT_s006
                                 + 0.877989 w_k1_MSFT_s006
                                 + 0.957916 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.923898 w_k1_ADM_s006
                                 + 1.19123 w_k1_AXP_s006
                                 + 1.28386 w_k1_DIS_s006
                                 + 1.05334 w_k1_FDX_s006
                                 + 1.72201 w_k1_IBM_s006
                                 + 1.15278 w_k1_INTC_s006
                                 + 1.34862 w_k1_KO_s006 + 1.623 w_k1_MAT_s006
                                 + 1.07278 w_k1_MSFT_s006
                                 + 1.25807 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.995946 w_k1_ADM_s007
                                 + 1.19361 w_k1_AXP_s007 + 1.1329 w_k1_DIS_s007
                                 + 1.21284 w_k1_FDX_s007
                                 + 0.760254 w_k1_IBM_s007
                                 + 1.3268 w_k1_INTC_s007
                                 + 0.868762 w_k1_KO_s007
                                 + 0.723062 w_k1_MAT_s007
                                 + 1.1957 w_k1_MSFT_s007
                                 + 0.995878 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.08104 w_k1_ADM_s007
                                 + 1.21184 w_k1_AXP_s007
                                 + 1.22535 w_k1_DIS_s007
                                 + 1.14171 w_k1_FDX_s007
                                 + 1.05638 w_k1_IBM_s007
                                 + 1.1104 w_k1_INTC_s007 + 1.12401 w_k1_KO_s007
                                 + 1.07613 w_k1_MAT_s007
                                 + 1.15487 w_k1_MSFT_s007
                                 + 1.12965 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.770434 w_k1_ADM_s007
                                 + 1.05549 w_k1_AXP_s007
                                 + 0.706111 w_k1_DIS_s007
                                 + 0.893144 w_k1_FDX_s007
                                 + 0.962451 w_k1_IBM_s007
                                 + 0.878936 w_k1_INTC_s007
                                 + 0.800502 w_k1_KO_s007
                                 + 0.966069 w_k1_MAT_s007
                                 + 0.805291 w_k1_MSFT_s007
                                 + 0.853398 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.997669 w_k1_ADM_s007
                                 + 1.15397 w_k1_AXP_s007
                                 + 1.27828 w_k1_DIS_s007
                                 + 0.931275 w_k1_FDX_s007
                                 + 2.03955 w_k1_IBM_s007
                                 + 0.918967 w_k1_INTC_s007
                                 + 1.20314 w_k1_KO_s007 + 1.55668 w_k1_MAT_s007
                                 + 1.00977 w_k1_MSFT_s007
                                 + 1.15469 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.721307 w_k1_ADM_s007
                                 + 0.674646 w_k1_AXP_s007
                                 + 0.596207 w_k1_DIS_s007
                                 + 0.756709 w_k1_FDX_s007
                                 + 0.63309 w_k1_IBM_s007
                                 + 0.721751 w_k1_INTC_s007
                                 + 0.619481 w_k1_KO_s007
                                 + 0.629386 w_k1_MAT_s007
                                 + 0.863843 w_k1_MSFT_s007
                                 + 0.749617 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.03107 w_k1_ADM_s007
                                 + 0.87933 w_k1_AXP_s007
                                 + 0.985504 w_k1_DIS_s007
                                 + 1.03024 w_k1_FDX_s007
                                 + 0.564389 w_k1_IBM_s007
                                 + 0.985691 w_k1_INTC_s007
                                 + 0.742321 w_k1_KO_s007
                                 + 0.508362 w_k1_MAT_s007
                                 + 0.969688 w_k1_MSFT_s007
                                 + 0.835429 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.08388 w_k1_ADM_s007
                                 + 1.133 w_k1_AXP_s007 + 1.19729 w_k1_DIS_s007
                                 + 1.0732 w_k1_FDX_s007 + 1.4268 w_k1_IBM_s007
                                 + 1.04087 w_k1_INTC_s007
                                 + 1.14077 w_k1_KO_s007 + 1.02106 w_k1_MAT_s007
                                 + 1.085 w_k1_MSFT_s007 + 1.12772 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.786439 w_k1_ADM_s007
                                 + 0.697531 w_k1_AXP_s007
                                 + 0.823382 w_k1_DIS_s007
                                 + 0.846345 w_k1_FDX_s007
                                 + 0.801071 w_k1_IBM_s007
                                 + 0.872977 w_k1_INTC_s007
                                 + 0.746236 w_k1_KO_s007
                                 + 0.430395 w_k1_MAT_s007
                                 + 0.941037 w_k1_MSFT_s007
                                 + 0.853912 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.16186 w_k1_ADM_s010
                                 + 1.2542 w_k1_AXP_s010 + 1.53515 w_k1_DIS_s010
                                 + 1.32609 w_k1_FDX_s010 + 1.0955 w_k1_IBM_s010
                                 + 1.22347 w_k1_INTC_s010
                                 + 1.05503 w_k1_KO_s010 + 1.20224 w_k1_MAT_s010
                                 + 1.3232 w_k1_MSFT_s010 + 1.0698 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.41211 w_k1_ADM_s010
                                 + 1.27583 w_k1_AXP_s010
                                 + 1.44838 w_k1_DIS_s010
                                 + 1.12322 w_k1_FDX_s010 + 1.6264 w_k1_IBM_s010
                                 + 1.18132 w_k1_INTC_s010 + 1.2162 w_k1_KO_s010
                                 + 1.41369 w_k1_MAT_s010
                                 + 1.34822 w_k1_MSFT_s010 + 1.1871 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.05735 w_k1_ADM_s010
                                 + 1.09588 w_k1_AXP_s010
                                 + 0.973435 w_k1_DIS_s010
                                 + 1.01252 w_k1_FDX_s010
                                 + 1.20872 w_k1_IBM_s010
                                 + 1.01328 w_k1_INTC_s010
                                 + 1.08537 w_k1_KO_s010 + 1.32201 w_k1_MAT_s010
                                 + 1.04761 w_k1_MSFT_s010
                                 + 1.03609 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.818651 w_k1_ADM_s010
                                 + 0.642407 w_k1_AXP_s010
                                 + 0.608198 w_k1_DIS_s010
                                 + 0.753883 w_k1_FDX_s010
                                 + 0.757055 w_k1_IBM_s010
                                 + 0.833005 w_k1_INTC_s010
                                 + 0.653097 w_k1_KO_s010
                                 + 0.755877 w_k1_MAT_s010
                                 + 0.811576 w_k1_MSFT_s010
                                 + 0.802131 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.753055 w_k1_ADM_s010
                                 + 0.565892 w_k1_AXP_s010
                                 + 0.781864 w_k1_DIS_s010
                                 + 0.858232 w_k1_FDX_s010
                                 + 0.918449 w_k1_IBM_s010
                                 + 0.908923 w_k1_INTC_s010
                                 + 0.846181 w_k1_KO_s010
                                 + 0.761202 w_k1_MAT_s010
                                 + 1.00458 w_k1_MSFT_s010
                                 + 0.933314 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.11467 w_k1_ADM_s010
                                 + 0.897956 w_k1_AXP_s010
                                 + 1.00456 w_k1_DIS_s010
                                 + 1.01917 w_k1_FDX_s010
                                 + 0.740538 w_k1_IBM_s010
                                 + 1.09274 w_k1_INTC_s010
                                 + 0.799833 w_k1_KO_s010
                                 + 0.684104 w_k1_MAT_s010
                                 + 1.06692 w_k1_MSFT_s010 + 0.9752 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.41425 w_k1_ADM_s010
                                 + 0.915503 w_k1_AXP_s010
                                 + 1.1973 w_k1_DIS_s010 + 1.18346 w_k1_FDX_s010
                                 + 0.860361 w_k1_IBM_s010
                                 + 1.0322 w_k1_INTC_s010
                                 + 0.976637 w_k1_KO_s010
                                 + 0.855216 w_k1_MAT_s010
                                 + 1.34585 w_k1_MSFT_s010
                                 + 0.957617 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.03023 w_k1_ADM_s010
                                 + 0.753077 w_k1_AXP_s010
                                 + 0.686659 w_k1_DIS_s010
                                 + 0.77757 w_k1_FDX_s010
                                 + 1.26571 w_k1_IBM_s010
                                 + 0.794773 w_k1_INTC_s010
                                 + 0.873727 w_k1_KO_s010
                                 + 1.08259 w_k1_MAT_s010
                                 + 0.915734 w_k1_MSFT_s010
                                 + 0.942282 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.19938 w_k1_ADM_s011
                                 + 1.23725 w_k1_AXP_s011
                                 + 1.18816 w_k1_DIS_s011
                                 + 1.20107 w_k1_FDX_s011
                                 + 1.08261 w_k1_IBM_s011
                                 + 1.21275 w_k1_INTC_s011
                                 + 1.14172 w_k1_KO_s011 + 1.49712 w_k1_MAT_s011
                                 + 1.07327 w_k1_MSFT_s011
                                 + 1.17306 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.968583 w_k1_ADM_s011
                                 + 1.14637 w_k1_AXP_s011
                                 + 1.05358 w_k1_DIS_s011
                                 + 1.21323 w_k1_FDX_s011
                                 + 0.807807 w_k1_IBM_s011
                                 + 1.28691 w_k1_INTC_s011
                                 + 0.956998 w_k1_KO_s011
                                 + 1.05647 w_k1_MAT_s011
                                 + 1.03413 w_k1_MSFT_s011
                                 + 0.95808 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.806586 w_k1_ADM_s011
                                 + 1.04747 w_k1_AXP_s011
                                 + 0.53222 w_k1_DIS_s011
                                 + 0.766163 w_k1_FDX_s011
                                 + 1.17839 w_k1_IBM_s011
                                 + 0.937703 w_k1_INTC_s011
                                 + 0.867843 w_k1_KO_s011
                                 + 1.67587 w_k1_MAT_s011
                                 + 0.723601 w_k1_MSFT_s011
                                 + 0.861418 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.35777 w_k1_ADM_s011
                                 + 1.66608 w_k1_AXP_s011 + 1.4781 w_k1_DIS_s011
                                 + 1.25765 w_k1_FDX_s011
                                 + 1.48978 w_k1_IBM_s011
                                 + 1.29897 w_k1_INTC_s011
                                 + 1.63223 w_k1_KO_s011 + 1.96576 w_k1_MAT_s011
                                 + 1.19727 w_k1_MSFT_s011
                                 + 1.37323 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.06572 w_k1_ADM_s011
                                 + 1.01214 w_k1_AXP_s011
                                 + 1.01429 w_k1_DIS_s011
                                 + 0.970979 w_k1_FDX_s011
                                 + 1.33577 w_k1_IBM_s011
                                 + 1.1335 w_k1_INTC_s011 + 1.30613 w_k1_KO_s011
                                 + 1.39729 w_k1_MAT_s011
                                 + 1.07004 w_k1_MSFT_s011
                                 + 1.06887 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.12109 w_k1_ADM_s011
                                 + 0.97309 w_k1_AXP_s011
                                 + 0.87941 w_k1_DIS_s011
                                 + 0.916387 w_k1_FDX_s011
                                 + 0.98141 w_k1_IBM_s011
                                 + 0.977244 w_k1_INTC_s011
                                 + 1.08266 w_k1_KO_s011 + 1.06387 w_k1_MAT_s011
                                 + 0.893518 w_k1_MSFT_s011
                                 + 0.995782 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.07527 w_k1_ADM_s011
                                 + 0.921617 w_k1_AXP_s011
                                 + 0.84224 w_k1_DIS_s011
                                 + 0.883492 w_k1_FDX_s011
                                 + 1.53332 w_k1_IBM_s011
                                 + 1.0604 w_k1_INTC_s011 + 1.30999 w_k1_KO_s011
                                 + 1.31671 w_k1_MAT_s011
                                 + 0.89032 w_k1_MSFT_s011
                                 + 0.93171 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.871176 w_k1_ADM_s011
                                 + 0.876343 w_k1_AXP_s011
                                 + 0.726422 w_k1_DIS_s011
                                 + 0.822903 w_k1_FDX_s011
                                 + 0.710854 w_k1_IBM_s011
                                 + 0.893102 w_k1_INTC_s011
                                 + 0.876411 w_k1_KO_s011
                                 + 0.887622 w_k1_MAT_s011
                                 + 0.802491 w_k1_MSFT_s011
                                 + 0.847485 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.01428 w_k1_ADM_s012
                                 + 0.967046 w_k1_AXP_s012
                                 + 1.11282 w_k1_DIS_s012
                                 + 1.12829 w_k1_FDX_s012
                                 + 1.01071 w_k1_IBM_s012
                                 + 0.938305 w_k1_INTC_s012
                                 + 1.03458 w_k1_KO_s012 + 1.35286 w_k1_MAT_s012
                                 + 1.07764 w_k1_MSFT_s012
                                 + 1.14617 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.24965 w_k1_ADM_s012
                                 + 1.47902 w_k1_AXP_s012
                                 + 1.35871 w_k1_DIS_s012
                                 + 1.39528 w_k1_FDX_s012
                                 + 1.31207 w_k1_IBM_s012
                                 + 1.3058 w_k1_INTC_s012 + 1.32926 w_k1_KO_s012
                                 + 1.82973 w_k1_MAT_s012
                                 + 1.31157 w_k1_MSFT_s012
                                 + 1.40213 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.856182 w_k1_ADM_s012
                                 + 1.11868 w_k1_AXP_s012
                                 + 0.772465 w_k1_DIS_s012
                                 + 0.821016 w_k1_FDX_s012
                                 + 1.40401 w_k1_IBM_s012
                                 + 0.82152 w_k1_INTC_s012 + 1.0155 w_k1_KO_s012
                                 + 1.51257 w_k1_MAT_s012
                                 + 0.801943 w_k1_MSFT_s012
                                 + 1.03587 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.768215 w_k1_ADM_s012
                                 + 0.751545 w_k1_AXP_s012
                                 + 0.459875 w_k1_DIS_s012
                                 + 0.733313 w_k1_FDX_s012
                                 + 0.764556 w_k1_IBM_s012
                                 + 0.731277 w_k1_INTC_s012
                                 + 0.720462 w_k1_KO_s012
                                 + 0.899917 w_k1_MAT_s012
                                 + 0.799032 w_k1_MSFT_s012
                                 + 0.869785 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.11693 w_k1_ADM_s012
                                 + 1.0957 w_k1_AXP_s012
                                 + 0.902881 w_k1_DIS_s012
                                 + 1.10962 w_k1_FDX_s012
                                 + 0.70976 w_k1_IBM_s012
                                 + 0.92247 w_k1_INTC_s012
                                 + 0.827516 w_k1_KO_s012
                                 + 0.862552 w_k1_MAT_s012
                                 + 1.03776 w_k1_MSFT_s012
                                 + 1.07292 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06287 w_k1_ADM_s012
                                 + 0.993475 w_k1_AXP_s012
                                 + 0.832402 w_k1_DIS_s012
                                 + 1.00612 w_k1_FDX_s012
                                 + 0.917743 w_k1_IBM_s012
                                 + 0.828436 w_k1_INTC_s012
                                 + 0.909183 w_k1_KO_s012
                                 + 0.993354 w_k1_MAT_s012
                                 + 0.965544 w_k1_MSFT_s012
                                 + 1.00775 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.00533 w_k1_ADM_s012
                                 + 1.1128 w_k1_AXP_s012 + 1.13238 w_k1_DIS_s012
                                 + 1.02718 w_k1_FDX_s012
                                 + 1.67406 w_k1_IBM_s012
                                 + 1.05454 w_k1_INTC_s012
                                 + 1.29176 w_k1_KO_s012 + 1.92907 w_k1_MAT_s012
                                 + 1.09275 w_k1_MSFT_s012 + 1.1276 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.838165 w_k1_ADM_s012
                                 + 0.699779 w_k1_AXP_s012
                                 + 0.7711 w_k1_DIS_s012 + 0.81477 w_k1_FDX_s012
                                 + 0.738307 w_k1_IBM_s012
                                 + 0.974046 w_k1_INTC_s012
                                 + 0.808186 w_k1_KO_s012
                                 + 0.783297 w_k1_MAT_s012
                                 + 0.874983 w_k1_MSFT_s012
                                 + 0.823336 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.12992 w_k1_ADM_s013
                                 + 1.4083 w_k1_AXP_s013 + 1.44897 w_k1_DIS_s013
                                 + 1.54006 w_k1_FDX_s013
                                 + 1.01893 w_k1_IBM_s013
                                 + 1.47948 w_k1_INTC_s013
                                 + 1.07579 w_k1_KO_s013 + 1.05047 w_k1_MAT_s013
                                 + 1.30228 w_k1_MSFT_s013
                                 + 1.16856 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.929939 w_k1_ADM_s013
                                 + 1.09241 w_k1_AXP_s013
                                 + 1.07723 w_k1_DIS_s013
                                 + 1.12159 w_k1_FDX_s013
                                 + 0.694133 w_k1_IBM_s013
                                 + 0.938349 w_k1_INTC_s013
                                 + 0.873869 w_k1_KO_s013
                                 + 0.849375 w_k1_MAT_s013
                                 + 0.909841 w_k1_MSFT_s013
                                 + 1.03354 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.864384 w_k1_ADM_s013
                                 + 0.922183 w_k1_AXP_s013
                                 + 1.27301 w_k1_DIS_s013
                                 + 1.00787 w_k1_FDX_s013
                                 + 1.63425 w_k1_IBM_s013
                                 + 0.803033 w_k1_INTC_s013
                                 + 1.28017 w_k1_KO_s013 + 1.43165 w_k1_MAT_s013
                                 + 0.811128 w_k1_MSFT_s013
                                 + 1.04489 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17443 w_k1_ADM_s013
                                 + 1.01977 w_k1_AXP_s013
                                 + 1.13309 w_k1_DIS_s013
                                 + 1.19155 w_k1_FDX_s013
                                 + 0.933514 w_k1_IBM_s013
                                 + 0.948454 w_k1_INTC_s013
                                 + 1.01265 w_k1_KO_s013
                                 + 0.728364 w_k1_MAT_s013
                                 + 0.875339 w_k1_MSFT_s013
                                 + 1.03922 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.732317 w_k1_ADM_s013
                                 + 0.926798 w_k1_AXP_s013
                                 + 1.23487 w_k1_DIS_s013
                                 + 1.03703 w_k1_FDX_s013
                                 + 1.25177 w_k1_IBM_s013
                                 + 1.0238 w_k1_INTC_s013 + 1.25327 w_k1_KO_s013
                                 + 0.935409 w_k1_MAT_s013
                                 + 0.962252 w_k1_MSFT_s013
                                 + 1.04988 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.85492 w_k1_ADM_s013
                                 + 0.694424 w_k1_AXP_s013
                                 + 0.947664 w_k1_DIS_s013
                                 + 0.887376 w_k1_FDX_s013
                                 + 0.983804 w_k1_IBM_s013
                                 + 0.856339 w_k1_INTC_s013
                                 + 1.00608 w_k1_KO_s013
                                 + 0.904913 w_k1_MAT_s013
                                 + 0.782945 w_k1_MSFT_s013
                                 + 0.908644 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.815165 w_k1_ADM_s013
                                 + 0.678349 w_k1_AXP_s013
                                 + 0.668743 w_k1_DIS_s013
                                 + 0.860597 w_k1_FDX_s013
                                 + 0.678799 w_k1_IBM_s013
                                 + 0.758063 w_k1_INTC_s013
                                 + 0.671955 w_k1_KO_s013
                                 + 0.629988 w_k1_MAT_s013
                                 + 0.74559 w_k1_MSFT_s013
                                 + 0.815746 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.887917 w_k1_ADM_s013
                                 + 1.2244 w_k1_AXP_s013 + 1.10455 w_k1_DIS_s013
                                 + 1.20271 w_k1_FDX_s013
                                 + 0.91096 w_k1_IBM_s013
                                 + 1.08828 w_k1_INTC_s013
                                 + 1.02639 w_k1_KO_s013
                                 + 0.895026 w_k1_MAT_s013
                                 + 0.93622 w_k1_MSFT_s013
                                 + 1.05633 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.991658 w_k1_ADM_s014
                                 + 0.890706 w_k1_AXP_s014
                                 + 0.809229 w_k1_DIS_s014
                                 + 0.815388 w_k1_FDX_s014
                                 + 1.16534 w_k1_IBM_s014
                                 + 0.88315 w_k1_INTC_s014
                                 + 0.963199 w_k1_KO_s014
                                 + 0.975551 w_k1_MAT_s014
                                 + 0.884859 w_k1_MSFT_s014
                                 + 0.982379 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.76342 w_k1_ADM_s014
                                 + 0.732687 w_k1_AXP_s014
                                 + 0.872337 w_k1_DIS_s014
                                 + 0.872617 w_k1_FDX_s014
                                 + 0.776894 w_k1_IBM_s014
                                 + 0.869133 w_k1_INTC_s014
                                 + 0.876301 w_k1_KO_s014
                                 + 0.587266 w_k1_MAT_s014
                                 + 0.865739 w_k1_MSFT_s014
                                 + 0.911213 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.91881 w_k1_ADM_s014
                                 + 0.678098 w_k1_AXP_s014
                                 + 0.674523 w_k1_DIS_s014
                                 + 0.67647 w_k1_FDX_s014
                                 + 0.93954 w_k1_IBM_s014
                                 + 0.832333 w_k1_INTC_s014
                                 + 0.899712 w_k1_KO_s014
                                 + 0.46992 w_k1_MAT_s014
                                 + 0.656402 w_k1_MSFT_s014
                                 + 0.9172 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.854021 w_k1_ADM_s014
                                 + 0.882159 w_k1_AXP_s014
                                 + 0.687404 w_k1_DIS_s014
                                 + 0.805366 w_k1_FDX_s014
                                 + 0.730919 w_k1_IBM_s014
                                 + 0.724901 w_k1_INTC_s014
                                 + 0.687995 w_k1_KO_s014
                                 + 0.548364 w_k1_MAT_s014
                                 + 0.6899 w_k1_MSFT_s014
                                 + 0.788679 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.08277 w_k1_ADM_s014
                                 + 1.36594 w_k1_AXP_s014
                                 + 1.22034 w_k1_DIS_s014
                                 + 1.06035 w_k1_FDX_s014
                                 + 1.49594 w_k1_IBM_s014
                                 + 0.858281 w_k1_INTC_s014
                                 + 1.29418 w_k1_KO_s014 + 1.42561 w_k1_MAT_s014
                                 + 0.968338 w_k1_MSFT_s014
                                 + 1.13385 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02595 w_k1_ADM_s014
                                 + 1.24541 w_k1_AXP_s014
                                 + 0.727563 w_k1_DIS_s014
                                 + 0.900265 w_k1_FDX_s014
                                 + 0.965802 w_k1_IBM_s014
                                 + 0.802808 w_k1_INTC_s014
                                 + 0.936527 w_k1_KO_s014
                                 + 1.03705 w_k1_MAT_s014
                                 + 0.802956 w_k1_MSFT_s014
                                 + 0.9514 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.17995 w_k1_ADM_s014
                                 + 1.42034 w_k1_AXP_s014
                                 + 1.29447 w_k1_DIS_s014
                                 + 1.27709 w_k1_FDX_s014
                                 + 0.763627 w_k1_IBM_s014
                                 + 0.972173 w_k1_INTC_s014
                                 + 0.867447 w_k1_KO_s014
                                 + 0.899653 w_k1_MAT_s014
                                 + 1.07843 w_k1_MSFT_s014
                                 + 1.14857 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.17483 w_k1_ADM_s014
                                 + 0.984964 w_k1_AXP_s014
                                 + 1.32473 w_k1_DIS_s014
                                 + 1.00972 w_k1_FDX_s014
                                 + 1.13277 w_k1_IBM_s014
                                 + 1.18802 w_k1_INTC_s014
                                 + 1.33969 w_k1_KO_s014
                                 + 0.882765 w_k1_MAT_s014
                                 + 1.14878 w_k1_MSFT_s014
                                 + 1.13062 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.865679 w_k1_ADM_s015
                                 + 1.02483 w_k1_AXP_s015
                                 + 0.912508 w_k1_DIS_s015
                                 + 0.973392 w_k1_FDX_s015
                                 + 1.31349 w_k1_IBM_s015
                                 + 0.953895 w_k1_INTC_s015
                                 + 0.915407 w_k1_KO_s015
                                 + 1.30576 w_k1_MAT_s015
                                 + 0.838304 w_k1_MSFT_s015
                                 + 0.997836 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.13302 w_k1_ADM_s015
                                 + 1.09701 w_k1_AXP_s015
                                 + 1.11918 w_k1_DIS_s015
                                 + 1.09596 w_k1_FDX_s015
                                 + 0.80815 w_k1_IBM_s015
                                 + 0.865854 w_k1_INTC_s015
                                 + 0.877499 w_k1_KO_s015
                                 + 0.929231 w_k1_MAT_s015
                                 + 0.955569 w_k1_MSFT_s015
                                 + 0.922457 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.13859 w_k1_ADM_s015
                                 + 0.811849 w_k1_AXP_s015
                                 + 1.05146 w_k1_DIS_s015
                                 + 0.959839 w_k1_FDX_s015
                                 + 1.05979 w_k1_IBM_s015
                                 + 0.891196 w_k1_INTC_s015
                                 + 0.939486 w_k1_KO_s015
                                 + 0.886177 w_k1_MAT_s015
                                 + 0.990391 w_k1_MSFT_s015
                                 + 0.943813 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.974564 w_k1_ADM_s015
                                 + 1.35578 w_k1_AXP_s015
                                 + 1.49749 w_k1_DIS_s015
                                 + 1.16275 w_k1_FDX_s015 + 1.484 w_k1_IBM_s015
                                 + 1.15254 w_k1_INTC_s015
                                 + 1.19363 w_k1_KO_s015 + 1.91857 w_k1_MAT_s015
                                 + 1.11547 w_k1_MSFT_s015
                                 + 1.14609 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.58077 w_k1_ADM_s015
                                 + 1.39263 w_k1_AXP_s015 + 1.8221 w_k1_DIS_s015
                                 + 1.49545 w_k1_FDX_s015
                                 + 1.20902 w_k1_IBM_s015
                                 + 1.40912 w_k1_INTC_s015 + 1.2479 w_k1_KO_s015
                                 + 1.03269 w_k1_MAT_s015
                                 + 1.26888 w_k1_MSFT_s015
                                 + 1.13015 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.761826 w_k1_ADM_s015
                                 + 0.705714 w_k1_AXP_s015
                                 + 0.625173 w_k1_DIS_s015
                                 + 0.766797 w_k1_FDX_s015
                                 + 0.804754 w_k1_IBM_s015
                                 + 0.925594 w_k1_INTC_s015
                                 + 0.699743 w_k1_KO_s015
                                 + 0.452454 w_k1_MAT_s015
                                 + 0.702609 w_k1_MSFT_s015
                                 + 0.788695 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.15956 w_k1_ADM_s015
                                 + 1.35506 w_k1_AXP_s015
                                 + 1.34895 w_k1_DIS_s015
                                 + 1.27062 w_k1_FDX_s015
                                 + 1.02964 w_k1_IBM_s015
                                 + 1.2044 w_k1_INTC_s015
                                 + 0.910855 w_k1_KO_s015
                                 + 0.923407 w_k1_MAT_s015
                                 + 1.04812 w_k1_MSFT_s015
                                 + 1.06344 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.809523 w_k1_ADM_s015
                                 + 0.903694 w_k1_AXP_s015
                                 + 0.767424 w_k1_DIS_s015
                                 + 0.828073 w_k1_FDX_s015
                                 + 0.83718 w_k1_IBM_s015
                                 + 0.940942 w_k1_INTC_s015
                                 + 0.81276 w_k1_KO_s015 + 1.05989 w_k1_MAT_s015
                                 + 0.832272 w_k1_MSFT_s015
                                 + 0.966019 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.967694 w_k1_ADM_s016
                                 + 1.02667 w_k1_AXP_s016
                                 + 1.33075 w_k1_DIS_s016
                                 + 1.11344 w_k1_FDX_s016
                                 + 1.32166 w_k1_IBM_s016
                                 + 0.81413 w_k1_INTC_s016
                                 + 1.27479 w_k1_KO_s016
                                 + 0.869701 w_k1_MAT_s016
                                 + 1.01164 w_k1_MSFT_s016
                                 + 1.07563 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.11342 w_k1_ADM_s016
                                 + 1.12561 w_k1_AXP_s016
                                 + 1.09334 w_k1_DIS_s016 + 1.1723 w_k1_FDX_s016
                                 + 1.00874 w_k1_IBM_s016
                                 + 0.988864 w_k1_INTC_s016
                                 + 0.967813 w_k1_KO_s016
                                 + 0.917177 w_k1_MAT_s016
                                 + 1.12203 w_k1_MSFT_s016
                                 + 0.878051 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.67947 w_k1_ADM_s016
                                 + 0.670541 w_k1_AXP_s016
                                 + 0.724761 w_k1_DIS_s016
                                 + 0.831362 w_k1_FDX_s016
                                 + 0.713143 w_k1_IBM_s016
                                 + 0.809742 w_k1_INTC_s016
                                 + 0.860968 w_k1_KO_s016
                                 + 0.79253 w_k1_MAT_s016
                                 + 0.909225 w_k1_MSFT_s016
                                 + 0.802606 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.04621 w_k1_ADM_s016
                                 + 0.812392 w_k1_AXP_s016
                                 + 1.01011 w_k1_DIS_s016
                                 + 0.924457 w_k1_FDX_s016
                                 + 1.73032 w_k1_IBM_s016
                                 + 0.764657 w_k1_INTC_s016
                                 + 1.47732 w_k1_KO_s016 + 1.72323 w_k1_MAT_s016
                                 + 0.868781 w_k1_MSFT_s016
                                 + 1.12399 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.20494 w_k1_ADM_s016
                                 + 1.37292 w_k1_AXP_s016
                                 + 1.53019 w_k1_DIS_s016
                                 + 1.59357 w_k1_FDX_s016
                                 + 0.883787 w_k1_IBM_s016
                                 + 1.14029 w_k1_INTC_s016
                                 + 1.14143 w_k1_KO_s016 + 1.31282 w_k1_MAT_s016
                                 + 1.21828 w_k1_MSFT_s016 + 1.1953 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.16712 w_k1_ADM_s016
                                 + 1.73307 w_k1_AXP_s016
                                 + 1.44101 w_k1_DIS_s016
                                 + 1.25947 w_k1_FDX_s016
                                 + 1.33756 w_k1_IBM_s016 + 0.964 w_k1_INTC_s016
                                 + 1.28999 w_k1_KO_s016 + 1.51275 w_k1_MAT_s016
                                 + 1.15792 w_k1_MSFT_s016
                                 + 1.11896 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.811959 w_k1_ADM_s016
                                 + 0.86268 w_k1_AXP_s016
                                 + 0.780122 w_k1_DIS_s016
                                 + 0.971961 w_k1_FDX_s016
                                 + 1.11867 w_k1_IBM_s016
                                 + 0.848219 w_k1_INTC_s016
                                 + 1.00043 w_k1_KO_s016 + 1.38857 w_k1_MAT_s016
                                 + 0.932924 w_k1_MSFT_s016
                                 + 0.986871 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.785402 w_k1_ADM_s016
                                 + 1.02913 w_k1_AXP_s016
                                 + 1.02047 w_k1_DIS_s016
                                 + 1.15096 w_k1_FDX_s016
                                 + 0.590119 w_k1_IBM_s016
                                 + 1.09042 w_k1_INTC_s016
                                 + 0.973008 w_k1_KO_s016
                                 + 0.738927 w_k1_MAT_s016
                                 + 1.08046 w_k1_MSFT_s016
                                 + 0.963706 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.16129 w_k1_ADM_s017
                                 + 1.10463 w_k1_AXP_s017
                                 + 0.979994 w_k1_DIS_s017
                                 + 1.00972 w_k1_FDX_s017
                                 + 1.07082 w_k1_IBM_s017
                                 + 1.10139 w_k1_INTC_s017
                                 + 1.13067 w_k1_KO_s017 + 1.18339 w_k1_MAT_s017
                                 + 1.02626 w_k1_MSFT_s017
                                 + 1.06406 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.684241 w_k1_ADM_s017
                                 + 0.773201 w_k1_AXP_s017
                                 + 0.724284 w_k1_DIS_s017
                                 + 0.791797 w_k1_FDX_s017
                                 + 0.99857 w_k1_IBM_s017
                                 + 1.03401 w_k1_INTC_s017
                                 + 0.948314 w_k1_KO_s017
                                 + 1.11967 w_k1_MAT_s017
                                 + 0.837002 w_k1_MSFT_s017
                                 + 0.880877 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.905675 w_k1_ADM_s017
                                 + 1.34036 w_k1_AXP_s017
                                 + 1.16941 w_k1_DIS_s017
                                 + 1.02909 w_k1_FDX_s017
                                 + 1.78437 w_k1_IBM_s017
                                 + 1.11594 w_k1_INTC_s017 + 1.5191 w_k1_KO_s017
                                 + 2.4232 w_k1_MAT_s017
                                 + 1.09125 w_k1_MSFT_s017
                                 + 1.23641 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.06589 w_k1_ADM_s017
                                 + 1.10504 w_k1_AXP_s017
                                 + 1.05585 w_k1_DIS_s017
                                 + 1.06318 w_k1_FDX_s017
                                 + 0.802187 w_k1_IBM_s017
                                 + 1.21142 w_k1_INTC_s017
                                 + 0.879567 w_k1_KO_s017
                                 + 1.23154 w_k1_MAT_s017
                                 + 1.12291 w_k1_MSFT_s017
                                 + 0.95996 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.06042 w_k1_ADM_s017
                                 + 0.713759 w_k1_AXP_s017
                                 + 0.822158 w_k1_DIS_s017
                                 + 0.912593 w_k1_FDX_s017
                                 + 0.751475 w_k1_IBM_s017
                                 + 0.866822 w_k1_INTC_s017
                                 + 0.922571 w_k1_KO_s017
                                 + 0.957581 w_k1_MAT_s017
                                 + 1.01746 w_k1_MSFT_s017
                                 + 0.93795 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.28175 w_k1_ADM_s017
                                 + 1.35932 w_k1_AXP_s017
                                 + 1.17569 w_k1_DIS_s017
                                 + 1.30579 w_k1_FDX_s017
                                 + 0.935996 w_k1_IBM_s017
                                 + 1.18725 w_k1_INTC_s017 + 1.1021 w_k1_KO_s017
                                 + 1.38407 w_k1_MAT_s017
                                 + 1.18547 w_k1_MSFT_s017
                                 + 1.15415 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.38475 w_k1_ADM_s017
                                 + 1.40555 w_k1_AXP_s017
                                 + 1.18085 w_k1_DIS_s017
                                 + 1.17091 w_k1_FDX_s017
                                 + 1.23041 w_k1_IBM_s017
                                 + 1.21504 w_k1_INTC_s017
                                 + 1.23058 w_k1_KO_s017 + 1.65151 w_k1_MAT_s017
                                 + 1.12807 w_k1_MSFT_s017
                                 + 1.16521 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.78316 w_k1_ADM_s017
                                 + 0.616732 w_k1_AXP_s017
                                 + 0.606961 w_k1_DIS_s017
                                 + 0.79449 w_k1_FDX_s017
                                 + 0.616618 w_k1_IBM_s017
                                 + 0.951691 w_k1_INTC_s017
                                 + 0.733649 w_k1_KO_s017
                                 + 0.659515 w_k1_MAT_s017
                                 + 0.852093 w_k1_MSFT_s017
                                 + 0.766054 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.866363 w_k1_ADM_s100
                                 + 0.820017 w_k1_AXP_s100
                                 + 0.874644 w_k1_DIS_s100
                                 + 0.910293 w_k1_FDX_s100
                                 + 0.840161 w_k1_IBM_s100
                                 + 1.23448 w_k1_INTC_s100
                                 + 0.877958 w_k1_KO_s100
                                 + 0.749363 w_k1_MAT_s100
                                 + 0.973233 w_k1_MSFT_s100
                                 + 0.935005 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.981314 w_k1_ADM_s100
                                 + 1.24519 w_k1_AXP_s100
                                 + 1.26127 w_k1_DIS_s100
                                 + 1.11083 w_k1_FDX_s100
                                 + 1.75889 w_k1_IBM_s100
                                 + 1.38134 w_k1_INTC_s100 + 1.3391 w_k1_KO_s100
                                 + 1.71448 w_k1_MAT_s100
                                 + 1.33106 w_k1_MSFT_s100
                                 + 1.07894 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.803415 w_k1_ADM_s100
                                 + 0.740256 w_k1_AXP_s100
                                 + 0.667861 w_k1_DIS_s100
                                 + 0.725574 w_k1_FDX_s100
                                 + 1.05564 w_k1_IBM_s100
                                 + 0.902869 w_k1_INTC_s100
                                 + 0.715593 w_k1_KO_s100
                                 + 1.03412 w_k1_MAT_s100
                                 + 0.794953 w_k1_MSFT_s100
                                 + 0.907846 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.783301 w_k1_ADM_s100
                                 + 0.819111 w_k1_AXP_s100
                                 + 0.85919 w_k1_DIS_s100
                                 + 0.988202 w_k1_FDX_s100
                                 + 0.714252 w_k1_IBM_s100
                                 + 1.19149 w_k1_INTC_s100
                                 + 0.756857 w_k1_KO_s100
                                 + 0.496105 w_k1_MAT_s100
                                 + 0.980832 w_k1_MSFT_s100
                                 + 0.847395 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.79256 w_k1_ADM_s100
                                 + 0.808847 w_k1_AXP_s100
                                 + 0.923686 w_k1_DIS_s100
                                 + 0.954989 w_k1_FDX_s100
                                 + 1.28969 w_k1_IBM_s100
                                 + 1.02225 w_k1_INTC_s100
                                 + 1.10949 w_k1_KO_s100 + 1.09755 w_k1_MAT_s100
                                 + 0.973761 w_k1_MSFT_s100
                                 + 1.04657 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.77132 w_k1_ADM_s100
                                 + 1.26547 w_k1_AXP_s100 + 1.2873 w_k1_DIS_s100
                                 + 1.32402 w_k1_FDX_s100
                                 + 0.963952 w_k1_IBM_s100
                                 + 1.38275 w_k1_INTC_s100
                                 + 0.949481 w_k1_KO_s100 + 1.0262 w_k1_MAT_s100
                                 + 1.11748 w_k1_MSFT_s100
                                 + 1.05886 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.889723 w_k1_ADM_s100
                                 + 1.29731 w_k1_AXP_s100 + 1.1175 w_k1_DIS_s100
                                 + 1.12834 w_k1_FDX_s100
                                 + 1.19343 w_k1_IBM_s100
                                 + 1.21004 w_k1_INTC_s100
                                 + 1.06325 w_k1_KO_s100 + 1.35986 w_k1_MAT_s100
                                 + 1.1619 w_k1_MSFT_s100 + 1.03047 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.956997 w_k1_ADM_s100
                                 + 1.10898 w_k1_AXP_s100
                                 + 1.55679 w_k1_DIS_s100
                                 + 1.19115 w_k1_FDX_s100
                                 + 1.09563 w_k1_IBM_s100
                                 + 1.29271 w_k1_INTC_s100
                                 + 1.22146 w_k1_KO_s100 + 1.15917 w_k1_MAT_s100
                                 + 1.28632 w_k1_MSFT_s100
                                 + 1.19838 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.00474 w_k1_ADM_s101
                                 + 1.21904 w_k1_AXP_s101
                                 + 1.10809 w_k1_DIS_s101
                                 + 1.01476 w_k1_FDX_s101
                                 + 0.974339 w_k1_IBM_s101
                                 + 1.11436 w_k1_INTC_s101
                                 + 0.912422 w_k1_KO_s101
                                 + 1.00226 w_k1_MAT_s101
                                 + 0.955909 w_k1_MSFT_s101
                                 + 1.02463 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.927268 w_k1_ADM_s101
                                 + 0.732526 w_k1_AXP_s101
                                 + 0.65787 w_k1_DIS_s101
                                 + 0.832399 w_k1_FDX_s101
                                 + 0.620356 w_k1_IBM_s101
                                 + 0.928063 w_k1_INTC_s101
                                 + 0.636762 w_k1_KO_s101
                                 + 0.585203 w_k1_MAT_s101
                                 + 0.741307 w_k1_MSFT_s101
                                 + 0.883219 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.655927 w_k1_ADM_s101
                                 + 0.728573 w_k1_AXP_s101
                                 + 0.597868 w_k1_DIS_s101
                                 + 0.650886 w_k1_FDX_s101
                                 + 0.883019 w_k1_IBM_s101
                                 + 0.946087 w_k1_INTC_s101
                                 + 0.784575 w_k1_KO_s101 + 0.8262 w_k1_MAT_s101
                                 + 0.762748 w_k1_MSFT_s101
                                 + 0.913284 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.15251 w_k1_ADM_s101
                                 + 1.14569 w_k1_AXP_s101
                                 + 1.33695 w_k1_DIS_s101
                                 + 1.22997 w_k1_FDX_s101
                                 + 0.824533 w_k1_IBM_s101
                                 + 1.09179 w_k1_INTC_s101
                                 + 1.05702 w_k1_KO_s101
                                 + 0.959381 w_k1_MAT_s101
                                 + 1.17263 w_k1_MSFT_s101
                                 + 1.18514 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.925562 w_k1_ADM_s101
                                 + 0.721476 w_k1_AXP_s101
                                 + 0.750093 w_k1_DIS_s101
                                 + 0.806357 w_k1_FDX_s101
                                 + 0.943564 w_k1_IBM_s101
                                 + 0.977937 w_k1_INTC_s101
                                 + 0.912165 w_k1_KO_s101
                                 + 0.79071 w_k1_MAT_s101
                                 + 0.86356 w_k1_MSFT_s101
                                 + 0.900444 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.934903 w_k1_ADM_s101
                                 + 1.08349 w_k1_AXP_s101
                                 + 0.915542 w_k1_DIS_s101
                                 + 0.906402 w_k1_FDX_s101
                                 + 1.2044 w_k1_IBM_s101
                                 + 1.02683 w_k1_INTC_s101
                                 + 1.00827 w_k1_KO_s101 + 1.30428 w_k1_MAT_s101
                                 + 0.938804 w_k1_MSFT_s101
                                 + 1.05093 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.810189 w_k1_ADM_s101
                                 + 0.709063 w_k1_AXP_s101
                                 + 0.919272 w_k1_DIS_s101
                                 + 0.909044 w_k1_FDX_s101
                                 + 0.665548 w_k1_IBM_s101
                                 + 1.17871 w_k1_INTC_s101
                                 + 0.822454 w_k1_KO_s101
                                 + 0.707723 w_k1_MAT_s101
                                 + 0.914603 w_k1_MSFT_s101
                                 + 0.948747 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.23214 w_k1_ADM_s101
                                 + 1.54047 w_k1_AXP_s101
                                 + 1.56456 w_k1_DIS_s101
                                 + 1.20341 w_k1_FDX_s101
                                 + 1.71424 w_k1_IBM_s101
                                 + 1.38791 w_k1_INTC_s101
                                 + 1.44922 w_k1_KO_s101 + 2.08296 w_k1_MAT_s101
                                 + 1.19682 w_k1_MSFT_s101
                                 + 1.35037 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.710995 w_k1_ADM_s102
                                 + 0.848016 w_k1_AXP_s102
                                 + 0.99808 w_k1_DIS_s102
                                 + 0.755793 w_k1_FDX_s102
                                 + 1.32299 w_k1_IBM_s102
                                 + 0.951701 w_k1_INTC_s102
                                 + 1.03937 w_k1_KO_s102 + 1.16154 w_k1_MAT_s102
                                 + 0.825865 w_k1_MSFT_s102
                                 + 0.85345 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.01804 w_k1_ADM_s102
                                 + 1.15018 w_k1_AXP_s102 + 1.0969 w_k1_DIS_s102
                                 + 1.07696 w_k1_FDX_s102
                                 + 1.04967 w_k1_IBM_s102
                                 + 0.983151 w_k1_INTC_s102
                                 + 0.91291 w_k1_KO_s102
                                 + 0.930507 w_k1_MAT_s102
                                 + 0.859858 w_k1_MSFT_s102
                                 + 1.00286 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.920902 w_k1_ADM_s102
                                 + 0.914374 w_k1_AXP_s102
                                 + 0.740385 w_k1_DIS_s102
                                 + 0.907423 w_k1_FDX_s102
                                 + 0.818825 w_k1_IBM_s102
                                 + 1.08557 w_k1_INTC_s102
                                 + 0.783988 w_k1_KO_s102
                                 + 0.787115 w_k1_MAT_s102
                                 + 0.844514 w_k1_MSFT_s102
                                 + 0.776932 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.57351 w_k1_ADM_s102
                                 + 0.723355 w_k1_AXP_s102
                                 + 0.537945 w_k1_DIS_s102
                                 + 0.686011 w_k1_FDX_s102
                                 + 0.97713 w_k1_IBM_s102
                                 + 0.885766 w_k1_INTC_s102
                                 + 0.663408 w_k1_KO_s102
                                 + 0.782767 w_k1_MAT_s102
                                 + 0.693312 w_k1_MSFT_s102
                                 + 0.762711 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.12466 w_k1_ADM_s102
                                 + 1.36907 w_k1_AXP_s102
                                 + 1.29802 w_k1_DIS_s102
                                 + 1.22208 w_k1_FDX_s102
                                 + 1.51094 w_k1_IBM_s102
                                 + 1.39116 w_k1_INTC_s102
                                 + 1.06073 w_k1_KO_s102 + 1.45584 w_k1_MAT_s102
                                 + 1.06663 w_k1_MSFT_s102
                                 + 1.06935 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04814 w_k1_ADM_s102
                                 + 0.919701 w_k1_AXP_s102
                                 + 0.858358 w_k1_DIS_s102
                                 + 0.897388 w_k1_FDX_s102
                                 + 1.17167 w_k1_IBM_s102
                                 + 1.08688 w_k1_INTC_s102
                                 + 0.917453 w_k1_KO_s102
                                 + 0.937057 w_k1_MAT_s102
                                 + 0.750535 w_k1_MSFT_s102
                                 + 0.891342 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.16826 w_k1_ADM_s102
                                 + 1.2599 w_k1_AXP_s102 + 1.51624 w_k1_DIS_s102
                                 + 1.3941 w_k1_FDX_s102
                                 + 0.996929 w_k1_IBM_s102
                                 + 1.43908 w_k1_INTC_s102
                                 + 1.06464 w_k1_KO_s102
                                 + 0.995925 w_k1_MAT_s102
                                 + 1.25879 w_k1_MSFT_s102
                                 + 1.05343 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.01622 w_k1_ADM_s102
                                 + 1.12603 w_k1_AXP_s102
                                 + 0.959681 w_k1_DIS_s102
                                 + 1.07785 w_k1_FDX_s102
                                 + 0.72718 w_k1_IBM_s102
                                 + 1.00473 w_k1_INTC_s102
                                 + 0.789617 w_k1_KO_s102
                                 + 0.777488 w_k1_MAT_s102
                                 + 0.918148 w_k1_MSFT_s102
                                 + 0.960045 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.06622 w_k1_ADM_s103
                                 + 0.883404 w_k1_AXP_s103
                                 + 0.994871 w_k1_DIS_s103
                                 + 1.05633 w_k1_FDX_s103
                                 + 0.786182 w_k1_IBM_s103
                                 + 1.1467 w_k1_INTC_s103
                                 + 0.917251 w_k1_KO_s103
                                 + 0.592378 w_k1_MAT_s103
                                 + 1.03048 w_k1_MSFT_s103
                                 + 1.07914 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.19072 w_k1_ADM_s103
                                 + 1.16292 w_k1_AXP_s103
                                 + 1.26863 w_k1_DIS_s103
                                 + 1.21208 w_k1_FDX_s103
                                 + 1.10502 w_k1_IBM_s103
                                 + 1.40903 w_k1_INTC_s103
                                 + 1.01993 w_k1_KO_s103 + 0.81827 w_k1_MAT_s103
                                 + 1.1781 w_k1_MSFT_s103 + 1.16336 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.851889 w_k1_ADM_s103
                                 + 0.754553 w_k1_AXP_s103
                                 + 0.68527 w_k1_DIS_s103
                                 + 0.770292 w_k1_FDX_s103
                                 + 0.832472 w_k1_IBM_s103
                                 + 0.748238 w_k1_INTC_s103
                                 + 0.844404 w_k1_KO_s103
                                 + 0.846916 w_k1_MAT_s103
                                 + 0.834237 w_k1_MSFT_s103
                                 + 0.894956 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.17215 w_k1_ADM_s103
                                 + 1.07106 w_k1_AXP_s103
                                 + 1.02825 w_k1_DIS_s103
                                 + 0.968648 w_k1_FDX_s103
                                 + 1.31223 w_k1_IBM_s103
                                 + 1.06528 w_k1_INTC_s103
                                 + 1.00905 w_k1_KO_s103 + 1.03713 w_k1_MAT_s103
                                 + 1.05366 w_k1_MSFT_s103
                                 + 0.941466 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.835042 w_k1_ADM_s103
                                 + 0.866641 w_k1_AXP_s103
                                 + 0.771223 w_k1_DIS_s103
                                 + 0.78098 w_k1_FDX_s103
                                 + 1.17838 w_k1_IBM_s103
                                 + 1.07553 w_k1_INTC_s103
                                 + 0.867129 w_k1_KO_s103
                                 + 1.07917 w_k1_MAT_s103
                                 + 0.914375 w_k1_MSFT_s103
                                 + 0.90652 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.24079 w_k1_ADM_s103
                                 + 1.24425 w_k1_AXP_s103
                                 + 1.24823 w_k1_DIS_s103
                                 + 1.03918 w_k1_FDX_s103 + 1.6568 w_k1_IBM_s103
                                 + 1.23112 w_k1_INTC_s103
                                 + 1.25607 w_k1_KO_s103 + 1.54089 w_k1_MAT_s103
                                 + 1.12471 w_k1_MSFT_s103
                                 + 1.19489 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.1115 w_k1_ADM_s103
                                 + 0.785438 w_k1_AXP_s103
                                 + 0.886867 w_k1_DIS_s103
                                 + 0.938968 w_k1_FDX_s103
                                 + 0.962567 w_k1_IBM_s103
                                 + 1.08314 w_k1_INTC_s103
                                 + 0.826622 w_k1_KO_s103
                                 + 0.523319 w_k1_MAT_s103
                                 + 0.939373 w_k1_MSFT_s103
                                 + 0.935997 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.35308 w_k1_ADM_s103
                                 + 1.54649 w_k1_AXP_s103
                                 + 1.81454 w_k1_DIS_s103
                                 + 1.40023 w_k1_FDX_s103
                                 + 1.51799 w_k1_IBM_s103
                                 + 1.23895 w_k1_INTC_s103
                                 + 1.37794 w_k1_KO_s103 + 1.72352 w_k1_MAT_s103
                                 + 1.19578 w_k1_MSFT_s103
                                 + 1.19562 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.702216 w_k1_ADM_s104
                                 + 0.734918 w_k1_AXP_s104
                                 + 0.518486 w_k1_DIS_s104
                                 + 0.817156 w_k1_FDX_s104
                                 + 0.629866 w_k1_IBM_s104
                                 + 0.944316 w_k1_INTC_s104
                                 + 0.622515 w_k1_KO_s104
                                 + 0.665459 w_k1_MAT_s104
                                 + 0.605489 w_k1_MSFT_s104
                                 + 0.848612 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.155 w_k1_ADM_s104
                                 + 1.00224 w_k1_AXP_s104
                                 + 1.29163 w_k1_DIS_s104
                                 + 1.37739 w_k1_FDX_s104
                                 + 0.837228 w_k1_IBM_s104
                                 + 1.39819 w_k1_INTC_s104
                                 + 1.15584 w_k1_KO_s104 + 1.16009 w_k1_MAT_s104
                                 + 1.33343 w_k1_MSFT_s104
                                 + 1.30392 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.872204 w_k1_ADM_s104
                                 + 0.741808 w_k1_AXP_s104
                                 + 0.830347 w_k1_DIS_s104
                                 + 0.875133 w_k1_FDX_s104
                                 + 0.753679 w_k1_IBM_s104
                                 + 1.12887 w_k1_INTC_s104
                                 + 0.894744 w_k1_KO_s104
                                 + 0.743071 w_k1_MAT_s104
                                 + 1.01155 w_k1_MSFT_s104
                                 + 0.92316 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.69251 w_k1_ADM_s104
                                 + 1.71505 w_k1_AXP_s104 + 1.5399 w_k1_DIS_s104
                                 + 1.3205 w_k1_FDX_s104 + 1.37212 w_k1_IBM_s104
                                 + 1.31421 w_k1_INTC_s104
                                 + 1.43824 w_k1_KO_s104 + 1.64307 w_k1_MAT_s104
                                 + 1.21606 w_k1_MSFT_s104
                                 + 1.29304 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.906246 w_k1_ADM_s104
                                 + 0.943018 w_k1_AXP_s104
                                 + 0.967451 w_k1_DIS_s104
                                 + 1.03244 w_k1_FDX_s104
                                 + 0.849454 w_k1_IBM_s104
                                 + 1.21936 w_k1_INTC_s104
                                 + 0.898096 w_k1_KO_s104
                                 + 0.951765 w_k1_MAT_s104
                                 + 0.935668 w_k1_MSFT_s104
                                 + 1.13926 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.23007 w_k1_ADM_s104
                                 + 1.35801 w_k1_AXP_s104
                                 + 1.63059 w_k1_DIS_s104
                                 + 1.28102 w_k1_FDX_s104
                                 + 1.63858 w_k1_IBM_s104
                                 + 1.38936 w_k1_INTC_s104
                                 + 1.42163 w_k1_KO_s104 + 2.39759 w_k1_MAT_s104
                                 + 1.2906 w_k1_MSFT_s104 + 1.30959 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.993795 w_k1_ADM_s104
                                 + 0.919372 w_k1_AXP_s104
                                 + 1.10201 w_k1_DIS_s104
                                 + 0.980409 w_k1_FDX_s104
                                 + 1.18672 w_k1_IBM_s104
                                 + 1.1492 w_k1_INTC_s104 + 1.17549 w_k1_KO_s104
                                 + 1.26636 w_k1_MAT_s104
                                 + 0.986618 w_k1_MSFT_s104
                                 + 1.01928 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.05652 w_k1_ADM_s104
                                 + 0.916221 w_k1_AXP_s104
                                 + 0.662507 w_k1_DIS_s104
                                 + 0.855805 w_k1_FDX_s104
                                 + 1.07006 w_k1_IBM_s104
                                 + 1.10697 w_k1_INTC_s104
                                 + 0.943519 w_k1_KO_s104
                                 + 1.32513 w_k1_MAT_s104
                                 + 0.800106 w_k1_MSFT_s104
                                 + 0.918694 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.861925 w_k1_ADM_s105
                                 + 1.05325 w_k1_AXP_s105
                                 + 1.06584 w_k1_DIS_s105
                                 + 1.06684 w_k1_FDX_s105
                                 + 1.20196 w_k1_IBM_s105
                                 + 0.857994 w_k1_INTC_s105
                                 + 1.11366 w_k1_KO_s105 + 1.37798 w_k1_MAT_s105
                                 + 0.924247 w_k1_MSFT_s105
                                 + 0.988357 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.885785 w_k1_ADM_s105
                                 + 0.87258 w_k1_AXP_s105
                                 + 0.672562 w_k1_DIS_s105
                                 + 0.791011 w_k1_FDX_s105
                                 + 0.966213 w_k1_IBM_s105
                                 + 0.85784 w_k1_INTC_s105
                                 + 0.996232 w_k1_KO_s105
                                 + 1.17935 w_k1_MAT_s105
                                 + 0.970682 w_k1_MSFT_s105
                                 + 0.904498 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.2619 w_k1_ADM_s105
                                 + 1.63832 w_k1_AXP_s105
                                 + 1.54747 w_k1_DIS_s105 + 1.4308 w_k1_FDX_s105
                                 + 1.40633 w_k1_IBM_s105
                                 + 1.02861 w_k1_INTC_s105
                                 + 1.39414 w_k1_KO_s105 + 1.37473 w_k1_MAT_s105
                                 + 1.16097 w_k1_MSFT_s105
                                 + 1.21148 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.10802 w_k1_ADM_s105
                                 + 0.970806 w_k1_AXP_s105
                                 + 1.27251 w_k1_DIS_s105
                                 + 1.15504 w_k1_FDX_s105
                                 + 0.914571 w_k1_IBM_s105
                                 + 0.922537 w_k1_INTC_s105
                                 + 1.12042 w_k1_KO_s105
                                 + 0.944651 w_k1_MAT_s105
                                 + 1.15719 w_k1_MSFT_s105
                                 + 0.948187 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.9152 w_k1_ADM_s105
                                 + 0.943982 w_k1_AXP_s105
                                 + 0.949099 w_k1_DIS_s105
                                 + 1.01088 w_k1_FDX_s105
                                 + 1.00998 w_k1_IBM_s105
                                 + 0.77539 w_k1_INTC_s105
                                 + 0.841377 w_k1_KO_s105
                                 + 1.25192 w_k1_MAT_s105
                                 + 0.92599 w_k1_MSFT_s105
                                 + 0.915898 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.06989 w_k1_ADM_s105
                                 + 1.17486 w_k1_AXP_s105
                                 + 0.837958 w_k1_DIS_s105
                                 + 1.11802 w_k1_FDX_s105
                                 + 0.884568 w_k1_IBM_s105
                                 + 0.905626 w_k1_INTC_s105
                                 + 0.859563 w_k1_KO_s105
                                 + 1.04795 w_k1_MAT_s105
                                 + 0.920937 w_k1_MSFT_s105
                                 + 0.912124 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.902056 w_k1_ADM_s105
                                 + 0.778267 w_k1_AXP_s105
                                 + 0.839762 w_k1_DIS_s105
                                 + 0.960884 w_k1_FDX_s105
                                 + 0.705465 w_k1_IBM_s105
                                 + 0.931684 w_k1_INTC_s105
                                 + 0.90869 w_k1_KO_s105
                                 + 0.690551 w_k1_MAT_s105
                                 + 0.923243 w_k1_MSFT_s105
                                 + 0.847938 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.71244 w_k1_ADM_s105
                                 + 0.710023 w_k1_AXP_s105
                                 + 0.651436 w_k1_DIS_s105
                                 + 0.765671 w_k1_FDX_s105
                                 + 0.663594 w_k1_IBM_s105
                                 + 0.662139 w_k1_INTC_s105
                                 + 0.776403 w_k1_KO_s105
                                 + 0.890307 w_k1_MAT_s105
                                 + 0.785378 w_k1_MSFT_s105
                                 + 0.80019 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.838456 w_k1_ADM_s106
                                 + 1.04472 w_k1_AXP_s106
                                 + 0.937136 w_k1_DIS_s106
                                 + 1.05057 w_k1_FDX_s106
                                 + 1.08679 w_k1_IBM_s106
                                 + 0.876778 w_k1_INTC_s106
                                 + 0.937831 w_k1_KO_s106
                                 + 0.924137 w_k1_MAT_s106
                                 + 0.916836 w_k1_MSFT_s106
                                 + 1.01552 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.10899 w_k1_ADM_s106
                                 + 1.1508 w_k1_AXP_s106 + 1.48381 w_k1_DIS_s106
                                 + 1.28998 w_k1_FDX_s106
                                 + 1.30582 w_k1_IBM_s106
                                 + 1.02672 w_k1_INTC_s106
                                 + 1.29863 w_k1_KO_s106 + 1.11459 w_k1_MAT_s106
                                 + 1.11799 w_k1_MSFT_s106
                                 + 1.32212 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.00608 w_k1_ADM_s106
                                 + 0.822979 w_k1_AXP_s106
                                 + 1.06808 w_k1_DIS_s106
                                 + 0.963979 w_k1_FDX_s106
                                 + 1.23034 w_k1_IBM_s106
                                 + 0.962199 w_k1_INTC_s106
                                 + 1.01931 w_k1_KO_s106 + 1.00319 w_k1_MAT_s106
                                 + 1.03396 w_k1_MSFT_s106
                                 + 1.04103 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.712163 w_k1_ADM_s106
                                 + 0.852395 w_k1_AXP_s106
                                 + 0.781782 w_k1_DIS_s106
                                 + 0.870025 w_k1_FDX_s106
                                 + 0.937208 w_k1_IBM_s106
                                 + 0.767493 w_k1_INTC_s106
                                 + 0.93727 w_k1_KO_s106
                                 + 0.932456 w_k1_MAT_s106
                                 + 0.842562 w_k1_MSFT_s106
                                 + 0.982428 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.860981 w_k1_ADM_s106
                                 + 0.877413 w_k1_AXP_s106
                                 + 1.02557 w_k1_DIS_s106
                                 + 0.994113 w_k1_FDX_s106
                                 + 0.756308 w_k1_IBM_s106
                                 + 0.915441 w_k1_INTC_s106
                                 + 0.850459 w_k1_KO_s106
                                 + 0.585251 w_k1_MAT_s106
                                 + 1.04505 w_k1_MSFT_s106
                                 + 0.982154 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.910358 w_k1_ADM_s106
                                 + 0.682023 w_k1_AXP_s106
                                 + 0.707095 w_k1_DIS_s106
                                 + 0.874427 w_k1_FDX_s106
                                 + 0.725629 w_k1_IBM_s106
                                 + 0.938039 w_k1_INTC_s106
                                 + 0.666695 w_k1_KO_s106
                                 + 0.521201 w_k1_MAT_s106
                                 + 0.900996 w_k1_MSFT_s106
                                 + 0.898997 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.855458 w_k1_ADM_s106
                                 + 0.93431 w_k1_AXP_s106
                                 + 1.28983 w_k1_DIS_s106
                                 + 1.00574 w_k1_FDX_s106
                                 + 1.39337 w_k1_IBM_s106
                                 + 0.976491 w_k1_INTC_s106
                                 + 1.32893 w_k1_KO_s106 + 1.29118 w_k1_MAT_s106
                                 + 1.07021 w_k1_MSFT_s106
                                 + 1.22201 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.15641 w_k1_ADM_s106
                                 + 1.13837 w_k1_AXP_s106
                                 + 1.22927 w_k1_DIS_s106
                                 + 1.31769 w_k1_FDX_s106
                                 + 0.747813 w_k1_IBM_s106
                                 + 1.14667 w_k1_INTC_s106
                                 + 0.821867 w_k1_KO_s106
                                 + 0.591664 w_k1_MAT_s106
                                 + 1.08829 w_k1_MSFT_s106
                                 + 1.01468 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.922342 w_k1_ADM_s107
                                 + 0.600179 w_k1_AXP_s107
                                 + 0.914329 w_k1_DIS_s107
                                 + 0.78662 w_k1_FDX_s107
                                 + 1.09522 w_k1_IBM_s107
                                 + 1.03929 w_k1_INTC_s107
                                 + 1.02845 w_k1_KO_s107
                                 + 0.950787 w_k1_MAT_s107
                                 + 0.995132 w_k1_MSFT_s107
                                 + 0.924484 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.35492 w_k1_ADM_s107
                                 + 1.32416 w_k1_AXP_s107
                                 + 1.53541 w_k1_DIS_s107
                                 + 1.33404 w_k1_FDX_s107
                                 + 1.10606 w_k1_IBM_s107
                                 + 1.01726 w_k1_INTC_s107 + 1.2794 w_k1_KO_s107
                                 + 1.15401 w_k1_MAT_s107 + 1.196 w_k1_MSFT_s107
                                 + 1.15566 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.866822 w_k1_ADM_s107
                                 + 0.728146 w_k1_AXP_s107
                                 + 0.726438 w_k1_DIS_s107
                                 + 0.796177 w_k1_FDX_s107
                                 + 0.914432 w_k1_IBM_s107
                                 + 0.778229 w_k1_INTC_s107
                                 + 0.816908 w_k1_KO_s107
                                 + 0.867395 w_k1_MAT_s107
                                 + 0.82041 w_k1_MSFT_s107
                                 + 0.85108 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.00294 w_k1_ADM_s107
                                 + 0.805706 w_k1_AXP_s107
                                 + 0.941661 w_k1_DIS_s107
                                 + 1.01287 w_k1_FDX_s107
                                 + 0.679383 w_k1_IBM_s107
                                 + 0.888398 w_k1_INTC_s107
                                 + 0.853449 w_k1_KO_s107
                                 + 0.71902 w_k1_MAT_s107
                                 + 0.969753 w_k1_MSFT_s107
                                 + 0.945212 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.28915 w_k1_ADM_s107
                                 + 1.17412 w_k1_AXP_s107
                                 + 1.71871 w_k1_DIS_s107 + 1.1648 w_k1_FDX_s107
                                 + 1.40536 w_k1_IBM_s107
                                 + 1.21018 w_k1_INTC_s107 + 1.6264 w_k1_KO_s107
                                 + 1.8396 w_k1_MAT_s107
                                 + 1.29634 w_k1_MSFT_s107
                                 + 1.28695 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.95318 w_k1_ADM_s107
                                 + 1.04831 w_k1_AXP_s107
                                 + 0.823497 w_k1_DIS_s107
                                 + 1.03814 w_k1_FDX_s107
                                 + 0.802202 w_k1_IBM_s107
                                 + 1.20391 w_k1_INTC_s107
                                 + 0.777967 w_k1_KO_s107 + 1.1906 w_k1_MAT_s107
                                 + 1.04998 w_k1_MSFT_s107
                                 + 1.00574 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.961438 w_k1_ADM_s107
                                 + 0.991715 w_k1_AXP_s107
                                 + 1.19112 w_k1_DIS_s107
                                 + 1.05888 w_k1_FDX_s107
                                 + 1.03014 w_k1_IBM_s107
                                 + 1.06974 w_k1_INTC_s107 + 1.1345 w_k1_KO_s107
                                 + 1.18548 w_k1_MAT_s107
                                 + 1.10188 w_k1_MSFT_s107
                                 + 1.19777 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.986097 w_k1_ADM_s107
                                 + 0.961071 w_k1_AXP_s107
                                 + 1.07547 w_k1_DIS_s107
                                 + 0.94207 w_k1_FDX_s107 + 1.3736 w_k1_IBM_s107
                                 + 1.0722 w_k1_INTC_s107 + 1.15812 w_k1_KO_s107
                                 + 1.70203 w_k1_MAT_s107
                                 + 1.03945 w_k1_MSFT_s107
                                 + 1.01987 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.868778 w_k1_ADM_s110
                                 + 0.972967 w_k1_AXP_s110
                                 + 0.704768 w_k1_DIS_s110
                                 + 0.75064 w_k1_FDX_s110
                                 + 1.09777 w_k1_IBM_s110
                                 + 0.813629 w_k1_INTC_s110
                                 + 0.869842 w_k1_KO_s110
                                 + 1.19054 w_k1_MAT_s110
                                 + 0.835877 w_k1_MSFT_s110
                                 + 0.958559 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.06714 w_k1_ADM_s110
                                 + 1.01758 w_k1_AXP_s110
                                 + 1.19894 w_k1_DIS_s110
                                 + 0.982435 w_k1_FDX_s110
                                 + 1.23975 w_k1_IBM_s110
                                 + 1.01633 w_k1_INTC_s110 + 1.1544 w_k1_KO_s110
                                 + 1.22598 w_k1_MAT_s110
                                 + 1.11853 w_k1_MSFT_s110
                                 + 1.12961 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.978244 w_k1_ADM_s110
                                 + 1.32382 w_k1_AXP_s110
                                 + 1.09661 w_k1_DIS_s110
                                 + 1.19788 w_k1_FDX_s110
                                 + 1.04508 w_k1_IBM_s110
                                 + 0.931337 w_k1_INTC_s110
                                 + 0.888437 w_k1_KO_s110
                                 + 1.46256 w_k1_MAT_s110
                                 + 0.994163 w_k1_MSFT_s110
                                 + 1.07188 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.24573 w_k1_ADM_s110
                                 + 1.28472 w_k1_AXP_s110
                                 + 1.72534 w_k1_DIS_s110
                                 + 1.34146 w_k1_FDX_s110
                                 + 1.11365 w_k1_IBM_s110
                                 + 1.23329 w_k1_INTC_s110
                                 + 1.37218 w_k1_KO_s110 + 1.17836 w_k1_MAT_s110
                                 + 1.34926 w_k1_MSFT_s110
                                 + 1.19764 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.688884 w_k1_ADM_s110
                                 + 0.673712 w_k1_AXP_s110
                                 + 0.825157 w_k1_DIS_s110
                                 + 0.776302 w_k1_FDX_s110
                                 + 0.884868 w_k1_IBM_s110
                                 + 0.75901 w_k1_INTC_s110
                                 + 0.776981 w_k1_KO_s110
                                 + 1.00688 w_k1_MAT_s110
                                 + 0.916027 w_k1_MSFT_s110
                                 + 0.905643 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.948601 w_k1_ADM_s110
                                 + 1.00174 w_k1_AXP_s110
                                 + 0.942668 w_k1_DIS_s110
                                 + 0.978788 w_k1_FDX_s110
                                 + 0.822913 w_k1_IBM_s110
                                 + 1.05912 w_k1_INTC_s110
                                 + 0.846139 w_k1_KO_s110
                                 + 0.904217 w_k1_MAT_s110
                                 + 1.04399 w_k1_MSFT_s110
                                 + 0.877764 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.10293 w_k1_ADM_s110
                                 + 1.07525 w_k1_AXP_s110
                                 + 1.08443 w_k1_DIS_s110
                                 + 1.02706 w_k1_FDX_s110
                                 + 0.938396 w_k1_IBM_s110
                                 + 0.841807 w_k1_INTC_s110
                                 + 0.967634 w_k1_KO_s110 + 1.0228 w_k1_MAT_s110
                                 + 1.06393 w_k1_MSFT_s110
                                 + 0.997559 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.900692 w_k1_ADM_s110
                                 + 0.801946 w_k1_AXP_s110
                                 + 0.638863 w_k1_DIS_s110
                                 + 0.812716 w_k1_FDX_s110
                                 + 0.662202 w_k1_IBM_s110
                                 + 0.724625 w_k1_INTC_s110
                                 + 0.707459 w_k1_KO_s110
                                 + 0.873892 w_k1_MAT_s110
                                 + 0.819786 w_k1_MSFT_s110
                                 + 0.805614 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.01143 w_k1_ADM_s111
                                 + 0.932921 w_k1_AXP_s111
                                 + 1.21107 w_k1_DIS_s111
                                 + 0.981265 w_k1_FDX_s111
                                 + 1.18343 w_k1_IBM_s111
                                 + 0.817463 w_k1_INTC_s111
                                 + 1.00275 w_k1_KO_s111
                                 + 0.989567 w_k1_MAT_s111
                                 + 0.934657 w_k1_MSFT_s111
                                 + 1.01241 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.12485 w_k1_ADM_s111
                                 + 1.24414 w_k1_AXP_s111
                                 + 1.54818 w_k1_DIS_s111
                                 + 1.31233 w_k1_FDX_s111
                                 + 1.17829 w_k1_IBM_s111
                                 + 1.24964 w_k1_INTC_s111
                                 + 1.24463 w_k1_KO_s111
                                 + 0.932519 w_k1_MAT_s111
                                 + 1.26874 w_k1_MSFT_s111
                                 + 1.13658 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.01389 w_k1_ADM_s111
                                 + 1.12246 w_k1_AXP_s111
                                 + 0.893231 w_k1_DIS_s111
                                 + 1.07979 w_k1_FDX_s111
                                 + 0.704426 w_k1_IBM_s111
                                 + 1.09791 w_k1_INTC_s111
                                 + 0.719867 w_k1_KO_s111
                                 + 1.05674 w_k1_MAT_s111
                                 + 0.983445 w_k1_MSFT_s111
                                 + 0.897224 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.933143 w_k1_ADM_s111
                                 + 0.906343 w_k1_AXP_s111
                                 + 0.897487 w_k1_DIS_s111
                                 + 0.97976 w_k1_FDX_s111
                                 + 1.07573 w_k1_IBM_s111
                                 + 1.02364 w_k1_INTC_s111
                                 + 1.05062 w_k1_KO_s111
                                 + 0.946957 w_k1_MAT_s111
                                 + 0.976699 w_k1_MSFT_s111
                                 + 1.00756 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.43431 w_k1_ADM_s111
                                 + 2.04019 w_k1_AXP_s111
                                 + 1.91865 w_k1_DIS_s111
                                 + 1.64243 w_k1_FDX_s111
                                 + 1.28371 w_k1_IBM_s111
                                 + 1.60033 w_k1_INTC_s111
                                 + 1.14381 w_k1_KO_s111 + 1.25895 w_k1_MAT_s111
                                 + 1.4261 w_k1_MSFT_s111 + 1.19327 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.849753 w_k1_ADM_s111
                                 + 0.718615 w_k1_AXP_s111
                                 + 0.639595 w_k1_DIS_s111
                                 + 0.759856 w_k1_FDX_s111
                                 + 0.842312 w_k1_IBM_s111
                                 + 0.627631 w_k1_INTC_s111
                                 + 0.642287 w_k1_KO_s111
                                 + 0.722777 w_k1_MAT_s111
                                 + 0.716243 w_k1_MSFT_s111 + 0.875 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.733196 w_k1_ADM_s111
                                 + 0.980975 w_k1_AXP_s111
                                 + 0.668909 w_k1_DIS_s111
                                 + 0.880134 w_k1_FDX_s111
                                 + 0.714401 w_k1_IBM_s111
                                 + 0.888078 w_k1_INTC_s111
                                 + 0.775095 w_k1_KO_s111
                                 + 0.888984 w_k1_MAT_s111
                                 + 0.89819 w_k1_MSFT_s111
                                 + 0.897215 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.998239 w_k1_ADM_s111
                                 + 1.42441 w_k1_AXP_s111
                                 + 1.17823 w_k1_DIS_s111
                                 + 1.23598 w_k1_FDX_s111
                                 + 1.49704 w_k1_IBM_s111
                                 + 0.959658 w_k1_INTC_s111
                                 + 1.17822 w_k1_KO_s111 + 1.93076 w_k1_MAT_s111
                                 + 0.995545 w_k1_MSFT_s111
                                 + 1.27893 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07299 w_k1_ADM_s112
                                 + 1.00558 w_k1_AXP_s112
                                 + 1.09525 w_k1_DIS_s112
                                 + 1.03519 w_k1_FDX_s112
                                 + 0.766191 w_k1_IBM_s112
                                 + 1.05316 w_k1_INTC_s112
                                 + 1.05844 w_k1_KO_s112
                                 + 0.585527 w_k1_MAT_s112
                                 + 1.06068 w_k1_MSFT_s112
                                 + 1.00426 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.787656 w_k1_ADM_s112
                                 + 0.922754 w_k1_AXP_s112
                                 + 0.960645 w_k1_DIS_s112
                                 + 0.879324 w_k1_FDX_s112
                                 + 1.15139 w_k1_IBM_s112
                                 + 1.04679 w_k1_INTC_s112
                                 + 1.02783 w_k1_KO_s112 + 1.02335 w_k1_MAT_s112
                                 + 1.02067 w_k1_MSFT_s112
                                 + 0.939359 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.920849 w_k1_ADM_s112
                                 + 1.1641 w_k1_AXP_s112 + 1.17787 w_k1_DIS_s112
                                 + 1.00077 w_k1_FDX_s112
                                 + 1.40182 w_k1_IBM_s112
                                 + 0.948526 w_k1_INTC_s112
                                 + 1.15796 w_k1_KO_s112 + 1.39407 w_k1_MAT_s112
                                 + 0.952012 w_k1_MSFT_s112
                                 + 1.0615 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.18132 w_k1_ADM_s112
                                 + 1.15455 w_k1_AXP_s112
                                 + 1.65773 w_k1_DIS_s112
                                 + 1.20659 w_k1_FDX_s112
                                 + 1.49859 w_k1_IBM_s112
                                 + 1.33417 w_k1_INTC_s112
                                 + 1.66388 w_k1_KO_s112 + 1.40701 w_k1_MAT_s112
                                 + 1.35767 w_k1_MSFT_s112
                                 + 1.20689 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.48159 w_k1_ADM_s112
                                 + 1.58428 w_k1_AXP_s112
                                 + 1.55078 w_k1_DIS_s112
                                 + 1.34203 w_k1_FDX_s112 + 1.0596 w_k1_IBM_s112
                                 + 1.35962 w_k1_INTC_s112
                                 + 1.22984 w_k1_KO_s112 + 1.23028 w_k1_MAT_s112
                                 + 1.24721 w_k1_MSFT_s112 + 1.1922 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.21834 w_k1_ADM_s112
                                 + 1.27526 w_k1_AXP_s112 + 1.0211 w_k1_DIS_s112
                                 + 1.0855 w_k1_FDX_s112
                                 + 0.913556 w_k1_IBM_s112
                                 + 1.00532 w_k1_INTC_s112
                                 + 0.919763 w_k1_KO_s112
                                 + 1.07845 w_k1_MAT_s112
                                 + 0.908051 w_k1_MSFT_s112
                                 + 0.952548 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.903275 w_k1_ADM_s112
                                 + 0.884887 w_k1_AXP_s112
                                 + 0.759759 w_k1_DIS_s112
                                 + 0.806151 w_k1_FDX_s112
                                 + 0.842961 w_k1_IBM_s112
                                 + 0.896063 w_k1_INTC_s112
                                 + 0.837985 w_k1_KO_s112
                                 + 1.01097 w_k1_MAT_s112
                                 + 0.824427 w_k1_MSFT_s112
                                 + 0.832788 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.2466 w_k1_ADM_s112
                                 + 1.32504 w_k1_AXP_s112
                                 + 1.14004 w_k1_DIS_s112
                                 + 1.23292 w_k1_FDX_s112
                                 + 1.23564 w_k1_IBM_s112
                                 + 1.06343 w_k1_INTC_s112
                                 + 1.04847 w_k1_KO_s112 + 1.48668 w_k1_MAT_s112
                                 + 1.08595 w_k1_MSFT_s112
                                 + 1.08384 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.890655 w_k1_ADM_s113
                                 + 0.676783 w_k1_AXP_s113
                                 + 0.885191 w_k1_DIS_s113
                                 + 0.844764 w_k1_FDX_s113
                                 + 0.922417 w_k1_IBM_s113
                                 + 0.944243 w_k1_INTC_s113
                                 + 1.07268 w_k1_KO_s113 + 1.46173 w_k1_MAT_s113
                                 + 1.00625 w_k1_MSFT_s113
                                 + 0.904905 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.782129 w_k1_ADM_s113
                                 + 0.90522 w_k1_AXP_s113
                                 + 0.827159 w_k1_DIS_s113
                                 + 0.800955 w_k1_FDX_s113
                                 + 1.22641 w_k1_IBM_s113
                                 + 0.856944 w_k1_INTC_s113
                                 + 1.05282 w_k1_KO_s113 + 2.07042 w_k1_MAT_s113
                                 + 0.786548 w_k1_MSFT_s113
                                 + 0.903672 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.03182 w_k1_ADM_s113
                                 + 0.895257 w_k1_AXP_s113
                                 + 0.903327 w_k1_DIS_s113
                                 + 1.03164 w_k1_FDX_s113
                                 + 0.843708 w_k1_IBM_s113
                                 + 0.877688 w_k1_INTC_s113
                                 + 1.14679 w_k1_KO_s113 + 1.32202 w_k1_MAT_s113
                                 + 1.04352 w_k1_MSFT_s113
                                 + 1.02246 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.933149 w_k1_ADM_s113
                                 + 0.744564 w_k1_AXP_s113
                                 + 0.570627 w_k1_DIS_s113
                                 + 0.78977 w_k1_FDX_s113
                                 + 0.627757 w_k1_IBM_s113
                                 + 0.73969 w_k1_INTC_s113
                                 + 0.696989 w_k1_KO_s113
                                 + 0.924584 w_k1_MAT_s113
                                 + 0.783882 w_k1_MSFT_s113
                                 + 0.796969 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.879157 w_k1_ADM_s113
                                 + 0.469042 w_k1_AXP_s113
                                 + 0.636712 w_k1_DIS_s113
                                 + 0.759692 w_k1_FDX_s113
                                 + 0.825442 w_k1_IBM_s113
                                 + 0.814237 w_k1_INTC_s113
                                 + 0.767628 w_k1_KO_s113
                                 + 0.851747 w_k1_MAT_s113
                                 + 0.798441 w_k1_MSFT_s113
                                 + 0.824945 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.12947 w_k1_ADM_s113
                                 + 0.900845 w_k1_AXP_s113
                                 + 1.44263 w_k1_DIS_s113
                                 + 1.14625 w_k1_FDX_s113
                                 + 1.19068 w_k1_IBM_s113
                                 + 1.17118 w_k1_INTC_s113
                                 + 1.34446 w_k1_KO_s113 + 1.64344 w_k1_MAT_s113
                                 + 1.20831 w_k1_MSFT_s113
                                 + 1.04551 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.33351 w_k1_ADM_s113
                                 + 1.31247 w_k1_AXP_s113
                                 + 1.34084 w_k1_DIS_s113
                                 + 1.32015 w_k1_FDX_s113
                                 + 1.06111 w_k1_IBM_s113
                                 + 1.30021 w_k1_INTC_s113
                                 + 1.34049 w_k1_KO_s113 + 1.74471 w_k1_MAT_s113
                                 + 1.26367 w_k1_MSFT_s113
                                 + 1.23339 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.797384 w_k1_ADM_s113
                                 + 0.648889 w_k1_AXP_s113
                                 + 0.555806 w_k1_DIS_s113
                                 + 0.678665 w_k1_FDX_s113
                                 + 1.24159 w_k1_IBM_s113
                                 + 0.794382 w_k1_INTC_s113
                                 + 0.960756 w_k1_KO_s113
                                 + 1.98882 w_k1_MAT_s113
                                 + 0.730468 w_k1_MSFT_s113
                                 + 0.87371 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.10411 w_k1_ADM_s114
                                 + 1.20632 w_k1_AXP_s114
                                 + 1.64167 w_k1_DIS_s114
                                 + 1.31751 w_k1_FDX_s114
                                 + 1.64932 w_k1_IBM_s114
                                 + 1.34302 w_k1_INTC_s114
                                 + 2.01305 w_k1_KO_s114 + 1.22357 w_k1_MAT_s114
                                 + 1.26027 w_k1_MSFT_s114
                                 + 1.25344 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.11258 w_k1_ADM_s114
                                 + 1.26072 w_k1_AXP_s114
                                 + 1.03011 w_k1_DIS_s114
                                 + 1.17209 w_k1_FDX_s114
                                 + 1.01832 w_k1_IBM_s114
                                 + 1.12318 w_k1_INTC_s114
                                 + 1.10821 w_k1_KO_s114 + 1.15934 w_k1_MAT_s114
                                 + 0.916829 w_k1_MSFT_s114
                                 + 1.13587 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.30478 w_k1_ADM_s114
                                 + 1.56163 w_k1_AXP_s114
                                 + 1.47302 w_k1_DIS_s114
                                 + 1.40629 w_k1_FDX_s114
                                 + 1.12615 w_k1_IBM_s114
                                 + 1.60243 w_k1_INTC_s114
                                 + 1.44271 w_k1_KO_s114 + 1.20837 w_k1_MAT_s114
                                 + 1.24399 w_k1_MSFT_s114
                                 + 1.24168 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.81857 w_k1_ADM_s114
                                 + 1.01194 w_k1_AXP_s114
                                 + 1.29768 w_k1_DIS_s114
                                 + 1.12052 w_k1_FDX_s114
                                 + 1.04159 w_k1_IBM_s114
                                 + 1.08073 w_k1_INTC_s114
                                 + 1.36877 w_k1_KO_s114 + 1.22344 w_k1_MAT_s114
                                 + 1.12597 w_k1_MSFT_s114
                                 + 1.06885 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.784574 w_k1_ADM_s114
                                 + 1.07033 w_k1_AXP_s114
                                 + 1.07462 w_k1_DIS_s114
                                 + 1.02892 w_k1_FDX_s114
                                 + 1.20515 w_k1_IBM_s114
                                 + 1.28109 w_k1_INTC_s114
                                 + 1.23379 w_k1_KO_s114
                                 + 0.923934 w_k1_MAT_s114
                                 + 1.02909 w_k1_MSFT_s114
                                 + 1.07323 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.924799 w_k1_ADM_s114
                                 + 0.96262 w_k1_AXP_s114
                                 + 0.744995 w_k1_DIS_s114
                                 + 0.896745 w_k1_FDX_s114
                                 + 0.818293 w_k1_IBM_s114
                                 + 0.902068 w_k1_INTC_s114
                                 + 0.909802 w_k1_KO_s114
                                 + 0.779779 w_k1_MAT_s114
                                 + 0.77233 w_k1_MSFT_s114
                                 + 0.904616 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.951495 w_k1_ADM_s114
                                 + 0.811169 w_k1_AXP_s114
                                 + 0.86938 w_k1_DIS_s114
                                 + 0.856179 w_k1_FDX_s114
                                 + 1.31888 w_k1_IBM_s114
                                 + 0.90383 w_k1_INTC_s114
                                 + 1.32519 w_k1_KO_s114 + 1.1498 w_k1_MAT_s114
                                 + 0.877655 w_k1_MSFT_s114
                                 + 1.02525 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.750468 w_k1_ADM_s114
                                 + 0.699962 w_k1_AXP_s114
                                 + 0.586165 w_k1_DIS_s114
                                 + 0.744674 w_k1_FDX_s114
                                 + 0.859825 w_k1_IBM_s114
                                 + 0.777675 w_k1_INTC_s114
                                 + 0.95483 w_k1_KO_s114
                                 + 0.704933 w_k1_MAT_s114
                                 + 0.639648 w_k1_MSFT_s114
                                 + 0.891365 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.37833 w_k1_ADM_s115
                                 + 1.20122 w_k1_AXP_s115
                                 + 1.41779 w_k1_DIS_s115
                                 + 1.38248 w_k1_FDX_s115
                                 + 0.904606 w_k1_IBM_s115
                                 + 1.17735 w_k1_INTC_s115
                                 + 1.10821 w_k1_KO_s115
                                 + 0.993331 w_k1_MAT_s115
                                 + 1.22438 w_k1_MSFT_s115
                                 + 1.14524 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.17153 w_k1_ADM_s115
                                 + 1.28686 w_k1_AXP_s115
                                 + 1.77115 w_k1_DIS_s115
                                 + 1.35473 w_k1_FDX_s115
                                 + 1.27196 w_k1_IBM_s115
                                 + 1.35966 w_k1_INTC_s115
                                 + 1.29504 w_k1_KO_s115 + 1.60446 w_k1_MAT_s115
                                 + 1.20708 w_k1_MSFT_s115
                                 + 1.31131 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.898496 w_k1_ADM_s115
                                 + 0.697632 w_k1_AXP_s115
                                 + 0.760965 w_k1_DIS_s115
                                 + 0.850743 w_k1_FDX_s115
                                 + 0.617101 w_k1_IBM_s115
                                 + 1.02979 w_k1_INTC_s115
                                 + 0.768639 w_k1_KO_s115
                                 + 0.597392 w_k1_MAT_s115
                                 + 0.942782 w_k1_MSFT_s115
                                 + 0.901346 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.16098 w_k1_ADM_s115
                                 + 1.11287 w_k1_AXP_s115 + 1.2358 w_k1_DIS_s115
                                 + 1.14219 w_k1_FDX_s115
                                 + 0.954167 w_k1_IBM_s115
                                 + 1.03555 w_k1_INTC_s115
                                 + 0.95112 w_k1_KO_s115 + 1.00011 w_k1_MAT_s115
                                 + 1.0444 w_k1_MSFT_s115 + 1.03527 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.942298 w_k1_ADM_s115
                                 + 0.662291 w_k1_AXP_s115
                                 + 0.720616 w_k1_DIS_s115
                                 + 0.731003 w_k1_FDX_s115
                                 + 1.08473 w_k1_IBM_s115
                                 + 0.902099 w_k1_INTC_s115
                                 + 0.864821 w_k1_KO_s115
                                 + 1.10069 w_k1_MAT_s115
                                 + 0.771721 w_k1_MSFT_s115
                                 + 0.879893 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.839737 w_k1_ADM_s115
                                 + 1.45662 w_k1_AXP_s115
                                 + 1.53369 w_k1_DIS_s115
                                 + 1.17787 w_k1_FDX_s115
                                 + 1.76776 w_k1_IBM_s115
                                 + 1.33991 w_k1_INTC_s115
                                 + 1.39421 w_k1_KO_s115 + 1.95246 w_k1_MAT_s115
                                 + 1.0722 w_k1_MSFT_s115 + 1.14433 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.17591 w_k1_ADM_s115
                                 + 0.925725 w_k1_AXP_s115
                                 + 0.987639 w_k1_DIS_s115
                                 + 0.965238 w_k1_FDX_s115
                                 + 1.39903 w_k1_IBM_s115
                                 + 1.12099 w_k1_INTC_s115
                                 + 1.21223 w_k1_KO_s115 + 1.17091 w_k1_MAT_s115
                                 + 0.964252 w_k1_MSFT_s115
                                 + 1.00811 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.959682 w_k1_ADM_s115
                                 + 0.75163 w_k1_AXP_s115
                                 + 0.857176 w_k1_DIS_s115
                                 + 0.852033 w_k1_FDX_s115
                                 + 0.819118 w_k1_IBM_s115
                                 + 1.0389 w_k1_INTC_s115
                                 + 0.878496 w_k1_KO_s115
                                 + 0.718484 w_k1_MAT_s115
                                 + 0.856293 w_k1_MSFT_s115
                                 + 0.862077 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.865415 w_k1_ADM_s116
                                 + 0.6062 w_k1_AXP_s116
                                 + 0.596221 w_k1_DIS_s116
                                 + 0.779696 w_k1_FDX_s116
                                 + 0.662659 w_k1_IBM_s116
                                 + 0.85953 w_k1_INTC_s116
                                 + 0.658584 w_k1_KO_s116
                                 + 0.517669 w_k1_MAT_s116
                                 + 0.844412 w_k1_MSFT_s116
                                 + 0.731628 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.1624 w_k1_ADM_s116
                                 + 1.30759 w_k1_AXP_s116
                                 + 0.843135 w_k1_DIS_s116
                                 + 1.14003 w_k1_FDX_s116
                                 + 0.810582 w_k1_IBM_s116
                                 + 1.25527 w_k1_INTC_s116
                                 + 0.77743 w_k1_KO_s116
                                 + 0.790172 w_k1_MAT_s116
                                 + 0.909089 w_k1_MSFT_s116
                                 + 0.924126 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.11043 w_k1_ADM_s116
                                 + 1.04858 w_k1_AXP_s116
                                 + 1.22578 w_k1_DIS_s116
                                 + 1.14768 w_k1_FDX_s116
                                 + 0.822224 w_k1_IBM_s116
                                 + 1.06135 w_k1_INTC_s116
                                 + 0.938545 w_k1_KO_s116
                                 + 0.82702 w_k1_MAT_s116
                                 + 0.965179 w_k1_MSFT_s116
                                 + 0.990837 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.1694 w_k1_ADM_s116
                                 + 0.927407 w_k1_AXP_s116
                                 + 0.797577 w_k1_DIS_s116
                                 + 0.878762 w_k1_FDX_s116
                                 + 0.924434 w_k1_IBM_s116
                                 + 0.80871 w_k1_INTC_s116
                                 + 0.893126 w_k1_KO_s116
                                 + 0.980796 w_k1_MAT_s116
                                 + 0.799939 w_k1_MSFT_s116
                                 + 0.840585 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.30282 w_k1_ADM_s116
                                 + 1.51267 w_k1_AXP_s116
                                 + 1.68797 w_k1_DIS_s116
                                 + 1.39334 w_k1_FDX_s116
                                 + 1.38324 w_k1_IBM_s116
                                 + 1.44444 w_k1_INTC_s116
                                 + 1.36509 w_k1_KO_s116 + 1.25772 w_k1_MAT_s116
                                 + 1.30321 w_k1_MSFT_s116
                                 + 1.15584 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.988076 w_k1_ADM_s116
                                 + 0.666735 w_k1_AXP_s116
                                 + 0.66896 w_k1_DIS_s116
                                 + 0.76103 w_k1_FDX_s116
                                 + 0.93039 w_k1_IBM_s116
                                 + 0.849316 w_k1_INTC_s116
                                 + 0.749978 w_k1_KO_s116
                                 + 0.800086 w_k1_MAT_s116
                                 + 0.730911 w_k1_MSFT_s116
                                 + 0.793501 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.13197 w_k1_ADM_s116
                                 + 0.751652 w_k1_AXP_s116
                                 + 0.929247 w_k1_DIS_s116
                                 + 1.00221 w_k1_FDX_s116
                                 + 0.687371 w_k1_IBM_s116
                                 + 0.916929 w_k1_INTC_s116
                                 + 0.880222 w_k1_KO_s116
                                 + 0.621795 w_k1_MAT_s116
                                 + 0.857494 w_k1_MSFT_s116
                                 + 0.838632 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.12975 w_k1_ADM_s116
                                 + 1.03759 w_k1_AXP_s116
                                 + 0.875166 w_k1_DIS_s116
                                 + 0.946693 w_k1_FDX_s116
                                 + 1.22328 w_k1_IBM_s116
                                 + 1.1747 w_k1_INTC_s116 + 1.05817 w_k1_KO_s116
                                 + 1.122 w_k1_MAT_s116
                                 + 0.955176 w_k1_MSFT_s116
                                 + 0.961304 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.28065 w_k1_ADM_s117
                                 + 1.05574 w_k1_AXP_s117
                                 + 1.47009 w_k1_DIS_s117
                                 + 1.29498 w_k1_FDX_s117
                                 + 0.922175 w_k1_IBM_s117
                                 + 1.15193 w_k1_INTC_s117
                                 + 1.21559 w_k1_KO_s117
                                 + 0.632064 w_k1_MAT_s117
                                 + 1.23387 w_k1_MSFT_s117
                                 + 1.03056 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.774702 w_k1_ADM_s117
                                 + 0.713781 w_k1_AXP_s117
                                 + 0.768287 w_k1_DIS_s117
                                 + 0.803335 w_k1_FDX_s117
                                 + 0.756521 w_k1_IBM_s117
                                 + 0.947292 w_k1_INTC_s117
                                 + 0.7931 w_k1_KO_s117 + 0.5245 w_k1_MAT_s117
                                 + 0.851023 w_k1_MSFT_s117
                                 + 0.886966 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.02209 w_k1_ADM_s117
                                 + 1.17255 w_k1_AXP_s117
                                 + 1.12209 w_k1_DIS_s117 + 1.0973 w_k1_FDX_s117
                                 + 0.927004 w_k1_IBM_s117
                                 + 0.999311 w_k1_INTC_s117
                                 + 1.07454 w_k1_KO_s117 + 0.70302 w_k1_MAT_s117
                                 + 0.995118 w_k1_MSFT_s117
                                 + 0.995673 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.832469 w_k1_ADM_s117
                                 + 0.951617 w_k1_AXP_s117
                                 + 0.888834 w_k1_DIS_s117
                                 + 0.911332 w_k1_FDX_s117
                                 + 0.825048 w_k1_IBM_s117
                                 + 0.937216 w_k1_INTC_s117
                                 + 0.849876 w_k1_KO_s117
                                 + 0.643262 w_k1_MAT_s117
                                 + 0.850071 w_k1_MSFT_s117
                                 + 0.896542 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.69272 w_k1_ADM_s117
                                 + 0.654046 w_k1_AXP_s117
                                 + 0.550257 w_k1_DIS_s117
                                 + 0.692522 w_k1_FDX_s117
                                 + 0.769516 w_k1_IBM_s117
                                 + 0.826189 w_k1_INTC_s117
                                 + 0.785953 w_k1_KO_s117
                                 + 0.627516 w_k1_MAT_s117
                                 + 0.676207 w_k1_MSFT_s117
                                 + 0.666458 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.696111 w_k1_ADM_s117
                                 + 0.877172 w_k1_AXP_s117
                                 + 0.793126 w_k1_DIS_s117
                                 + 0.786628 w_k1_FDX_s117
                                 + 1.32621 w_k1_IBM_s117
                                 + 0.8492 w_k1_INTC_s117 + 1.01646 w_k1_KO_s117
                                 + 1.23524 w_k1_MAT_s117
                                 + 0.807711 w_k1_MSFT_s117
                                 + 0.877631 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.21118 w_k1_ADM_s117
                                 + 1.55738 w_k1_AXP_s117
                                 + 1.37104 w_k1_DIS_s117
                                 + 1.28567 w_k1_FDX_s117
                                 + 1.44959 w_k1_IBM_s117
                                 + 1.24142 w_k1_INTC_s117
                                 + 1.43731 w_k1_KO_s117 + 1.77183 w_k1_MAT_s117
                                 + 1.05035 w_k1_MSFT_s117
                                 + 1.14014 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.04606 w_k1_ADM_s117
                                 + 1.08461 w_k1_AXP_s117
                                 + 1.33796 w_k1_DIS_s117
                                 + 1.10305 w_k1_FDX_s117
                                 + 1.36899 w_k1_IBM_s117
                                 + 1.04551 w_k1_INTC_s117
                                 + 1.29018 w_k1_KO_s117 + 1.03489 w_k1_MAT_s117
                                 + 1.05035 w_k1_MSFT_s117
                                 + 1.06442 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0217779043094664 w_k1_ADM_s000
                                 + 0.0354037901499437 w_k1_ADM_s001
                                 + 0.0159470001081931 w_k1_ADM_s002
                                 + 0.0408494127234851 w_k1_ADM_s003
                                 + 0.0317231495185756 w_k1_ADM_s004
                                 + 0.0117614874605086 w_k1_ADM_s005
                                 + 0.0170025993310874 w_k1_ADM_s006
                                 + 0.0479563589619778 w_k1_ADM_s007
                                 + 0.0187494495096439 w_k1_ADM_s010
                                 + 0.0184937198659254 w_k1_ADM_s011
                                 + 0.0248142469981196 w_k1_ADM_s012
                                 + 0.0069414729547805 w_k1_ADM_s013
                                 + 0.0279197118276732 w_k1_ADM_s014
                                 + 0.0183124479110965 w_k1_ADM_s015
                                 + 0.00625324930751839 w_k1_ADM_s016
                                 + 0.024687934003751 w_k1_ADM_s017
                                 + 0.0164280537229301 w_k1_ADM_s100
                                 + 0.0338852703353536 w_k1_ADM_s101
                                 + 0.012881110359617 w_k1_ADM_s102
                                 + 0.0238842298730183 w_k1_ADM_s103
                                 + 0.0105138449267045 w_k1_ADM_s104
                                 + 0.012087613166862 w_k1_ADM_s105
                                 + 0.0127456921584537 w_k1_ADM_s106
                                 + 0.0202184320299276 w_k1_ADM_s107
                                 + 0.0369745703846914 w_k1_ADM_s110
                                 + 0.040540104923625 w_k1_ADM_s111
                                 + 0.0951111369885449 w_k1_ADM_s112
                                 + 0.070355511527446 w_k1_ADM_s113
                                 + 0.0378803518055207 w_k1_ADM_s114
                                 + 0.0671083123119244 w_k1_ADM_s115
                                 + 0.0506790447396979 w_k1_ADM_s116
                                 + 0.108532966685098 w_k1_ADM_s117
                                 + 0.0193107482827996 w_k1_AXP_s000
                                 + 0.0328517560709045 w_k1_AXP_s001
                                 + 0.0128606866541417 w_k1_AXP_s002
                                 + 0.0412407678047301 w_k1_AXP_s003
                                 + 0.0282979700586094 w_k1_AXP_s004
                                 + 0.0116237371388723 w_k1_AXP_s005
                                 + 0.018447344001218 w_k1_AXP_s006
                                 + 0.0516053231528867 w_k1_AXP_s007
                                 + 0.0168875113065491 w_k1_AXP_s010
                                 + 0.0188505716331238 w_k1_AXP_s011
                                 + 0.0258351566202451 w_k1_AXP_s012
                                 + 0.00726374411837496 w_k1_AXP_s013
                                 + 0.0285853712402688 w_k1_AXP_s014
                                 + 0.0181035912457447 w_k1_AXP_s015
                                 + 0.00691318249767007 w_k1_AXP_s016
                                 + 0.0248195634740861 w_k1_AXP_s017
                                 + 0.0188848897171244 w_k1_AXP_s100
                                 + 0.0354692935397641 w_k1_AXP_s101
                                 + 0.014092427699307 w_k1_AXP_s102
                                 + 0.0215476853216689 w_k1_AXP_s103
                                 + 0.0100849344175744 w_k1_AXP_s104
                                 + 0.0130744817979418 w_k1_AXP_s105
                                 + 0.0128513290606563 w_k1_AXP_s106
                                 + 0.0181880451712409 w_k1_AXP_s107
                                 + 0.0382382663606506 w_k1_AXP_s110
                                 + 0.0434864541303487 w_k1_AXP_s111
                                 + 0.0991661082230535 w_k1_AXP_s112
                                 + 0.0614690625501712 w_k1_AXP_s113
                                 + 0.0410362375943987 w_k1_AXP_s114
                                 + 0.060832518137575 w_k1_AXP_s115
                                 + 0.0468485236118604 w_k1_AXP_s116
                                 + 0.120109721498579 w_k1_AXP_s117
                                 + 0.0201186300486486 w_k1_DIS_s000
                                 + 0.0322455908132766 w_k1_DIS_s001
                                 + 0.0129257904019683 w_k1_DIS_s002
                                 + 0.0418194972476863 w_k1_DIS_s003
                                 + 0.029925624578059 w_k1_DIS_s004
                                 + 0.013616303129371 w_k1_DIS_s005
                                 + 0.0187261305236404 w_k1_DIS_s006
                                 + 0.0514531562003409 w_k1_DIS_s007
                                 + 0.0187462083646834 w_k1_DIS_s010
                                 + 0.0167727342678831 w_k1_DIS_s011
                                 + 0.023069122389053 w_k1_DIS_s012
                                 + 0.00842904205362665 w_k1_DIS_s013
                                 + 0.0285610275870381 w_k1_DIS_s014
                                 + 0.0192549244394281 w_k1_DIS_s015
                                 + 0.00710379837501734 w_k1_DIS_s016
                                 + 0.0225185605104122 w_k1_DIS_s017
                                 + 0.0200216285551835 w_k1_DIS_s100
                                 + 0.0349237491470951 w_k1_DIS_s101
                                 + 0.0135038407238979 w_k1_DIS_s102
                                 + 0.0226147812441569 w_k1_DIS_s103
                                 + 0.0103646533902013 w_k1_DIS_s104
                                 + 0.013019927930092 w_k1_DIS_s105
                                 + 0.0143967364484551 w_k1_DIS_s106
                                 + 0.0211589852346888 w_k1_DIS_s107
                                 + 0.0398614841737849 w_k1_DIS_s110
                                 + 0.043583790475239 w_k1_DIS_s111
                                 + 0.0971692222891369 w_k1_DIS_s112
                                 + 0.0709818173308452 w_k1_DIS_s113
                                 + 0.0397993844563919 w_k1_DIS_s114
                                 + 0.0687880460739848 w_k1_DIS_s115
                                 + 0.0459845654664734 w_k1_DIS_s116
                                 + 0.121433617243163 w_k1_DIS_s117
                                 + 0.020457260840254 w_k1_FDX_s000
                                 + 0.0331684710763385 w_k1_FDX_s001
                                 + 0.0125779666518654 w_k1_FDX_s002
                                 + 0.0404286746079496 w_k1_FDX_s003
                                 + 0.0279475644592943 w_k1_FDX_s004
                                 + 0.0125051194624052 w_k1_FDX_s005
                                 + 0.017102874329592 w_k1_FDX_s006
                                 + 0.050805540423027 w_k1_FDX_s007
                                 + 0.0182320947567354 w_k1_FDX_s010
                                 + 0.0175546669665986 w_k1_FDX_s011
                                 + 0.0254152906070532 w_k1_FDX_s012
                                 + 0.00819651729871142 w_k1_FDX_s013
                                 + 0.0261967431121603 w_k1_FDX_s014
                                 + 0.0181932832156464 w_k1_FDX_s015
                                 + 0.00748699302588323 w_k1_FDX_s016
                                 + 0.0237465352699084 w_k1_FDX_s017
                                 + 0.0200403130099332 w_k1_FDX_s100
                                 + 0.0334663227954782 w_k1_FDX_s101
                                 + 0.0136932852611128 w_k1_FDX_s102
                                 + 0.0223079744403667 w_k1_FDX_s103
                                 + 0.0104532326302947 w_k1_FDX_s104
                                 + 0.0132904788682476 w_k1_FDX_s105
                                 + 0.0143481536104381 w_k1_FDX_s106
                                 + 0.0195849513912668 w_k1_FDX_s107
                                 + 0.0370976226827897 w_k1_FDX_s110
                                 + 0.0432689725916423 w_k1_FDX_s111
                                 + 0.0912037423223209 w_k1_FDX_s112
                                 + 0.06858627949451 w_k1_FDX_s113
                                 + 0.0407615005437012 w_k1_FDX_s114
                                 + 0.064518649654973 w_k1_FDX_s115
                                 + 0.0463636962395878 w_k1_FDX_s116
                                 + 0.115902448879365 w_k1_FDX_s117
                                 + 0.020155241695716 w_k1_IBM_s000
                                 + 0.0337834396600192 w_k1_IBM_s001
                                 + 0.0129620250033888 w_k1_IBM_s002
                                 + 0.0440594917244145 w_k1_IBM_s003
                                 + 0.0291470220471747 w_k1_IBM_s004
                                 + 0.0131569140629434 w_k1_IBM_s005
                                 + 0.0181656898855583 w_k1_IBM_s006
                                 + 0.0514719773246772 w_k1_IBM_s007
                                 + 0.0180000713503358 w_k1_IBM_s010
                                 + 0.0181156133114385 w_k1_IBM_s011
                                 + 0.0256181144198237 w_k1_IBM_s012
                                 + 0.00801549573822083 w_k1_IBM_s013
                                 + 0.0273682858420303 w_k1_IBM_s014
                                 + 0.0188800770222149 w_k1_IBM_s015
                                 + 0.00677911413833053 w_k1_IBM_s016
                                 + 0.0229695570511983 w_k1_IBM_s017
                                 + 0.0204169710186107 w_k1_IBM_s100
                                 + 0.0341534046902463 w_k1_IBM_s101
                                 + 0.0131758055393865 w_k1_IBM_s102
                                 + 0.0232610139478334 w_k1_IBM_s103
                                 + 0.0103496025491442 w_k1_IBM_s104
                                 + 0.0129302167662618 w_k1_IBM_s105
                                 + 0.0137991685281098 w_k1_IBM_s106
                                 + 0.0197738078410161 w_k1_IBM_s107
                                 + 0.036243271932429 w_k1_IBM_s110
                                 + 0.042563439408512 w_k1_IBM_s111
                                 + 0.0922445812355293 w_k1_IBM_s112
                                 + 0.0696904023820271 w_k1_IBM_s113
                                 + 0.0437567965051553 w_k1_IBM_s114
                                 + 0.0636867560674648 w_k1_IBM_s115
                                 + 0.0461114159732747 w_k1_IBM_s116
                                 + 0.123118061197522 w_k1_IBM_s117
                                 + 0.0209378344334293 w_k1_INTC_s000
                                 + 0.0323947708772181 w_k1_INTC_s001
                                 + 0.0145090275793808 w_k1_INTC_s002
                                 + 0.0391071967693152 w_k1_INTC_s003
                                 + 0.0246229771427604 w_k1_INTC_s004
                                 + 0.0129871690031741 w_k1_INTC_s005
                                 + 0.0193502069709436 w_k1_INTC_s006
                                 + 0.0505592241023953 w_k1_INTC_s007
                                 + 0.0181910817288201 w_k1_INTC_s010
                                 + 0.0188877466094277 w_k1_INTC_s011
                                 + 0.023118330367636 w_k1_INTC_s012
                                 + 0.00734037202059796 w_k1_INTC_s013
                                 + 0.0251628681490903 w_k1_INTC_s014
                                 + 0.0175719043564922 w_k1_INTC_s015
                                 + 0.0064170658399338 w_k1_INTC_s016
                                 + 0.0257377937044696 w_k1_INTC_s017
                                 + 0.0234461556528186 w_k1_INTC_s100
                                 + 0.0372583690731716 w_k1_INTC_s101
                                 + 0.014513578849954 w_k1_INTC_s102
                                 + 0.0246682650405516 w_k1_INTC_s103
                                 + 0.0120322118386106 w_k1_INTC_s104
                                 + 0.0108938719601834 w_k1_INTC_s105
                                 + 0.0128648864287003 w_k1_INTC_s106
                                 + 0.0191815505761625 w_k1_INTC_s107
                                 + 0.0349099532594533 w_k1_INTC_s110
                                 + 0.0406598375513639 w_k1_INTC_s111
                                 + 0.0944378698506416 w_k1_INTC_s112
                                 + 0.069024546918025 w_k1_INTC_s113
                                 + 0.0429056847841198 w_k1_INTC_s114
                                 + 0.0679770908743396 w_k1_INTC_s115
                                 + 0.0492551055774699 w_k1_INTC_s116
                                 + 0.115196297026181 w_k1_INTC_s117
                                 + 0.0207161405374925 w_k1_KO_s000
                                 + 0.0348486054298647 w_k1_KO_s001
                                 + 0.0138383014207637 w_k1_KO_s002
                                 + 0.0449082033349018 w_k1_KO_s003
                                 + 0.0283893054489659 w_k1_KO_s004
                                 + 0.0143430949152841 w_k1_KO_s005
                                 + 0.017146911435221 w_k1_KO_s006
                                 + 0.0470976272006687 w_k1_KO_s007
                                 + 0.0166023923885954 w_k1_KO_s010
                                 + 0.0192005544466295 w_k1_KO_s011
                                 + 0.0243112005952273 w_k1_KO_s012
                                 + 0.00795719394892281 w_k1_KO_s013
                                 + 0.0278881930478955 w_k1_KO_s014
                                 + 0.0165466253552783 w_k1_KO_s015
                                 + 0.00730645402937438 w_k1_KO_s016
                                 + 0.0240401206991814 w_k1_KO_s017
                                 + 0.0188793898889035 w_k1_KO_s100
                                 + 0.0330862549777408 w_k1_KO_s101
                                 + 0.0116381041927134 w_k1_KO_s102
                                 + 0.0213499535191009 w_k1_KO_s103
                                 + 0.010488918001398 w_k1_KO_s104
                                 + 0.013107836067448 w_k1_KO_s105
                                 + 0.0132474436410452 w_k1_KO_s106
                                 + 0.0205601450596149 w_k1_KO_s107
                                 + 0.0360404133870083 w_k1_KO_s110
                                 + 0.0395741331445127 w_k1_KO_s111
                                 + 0.0944470746195408 w_k1_KO_s112
                                 + 0.077674927289443 w_k1_KO_s113
                                 + 0.0489617033660417 w_k1_KO_s114
                                 + 0.0625768477057638 w_k1_KO_s115
                                 + 0.0442982076598978 w_k1_KO_s116
                                 + 0.122834469933285 w_k1_KO_s117
                                 + 0.0202489224243226 w_k1_MAT_s000
                                 + 0.0314551391947391 w_k1_MAT_s001
                                 + 0.0148491543510258 w_k1_MAT_s002
                                 + 0.0464509490510051 w_k1_MAT_s003
                                 + 0.0342392652852031 w_k1_MAT_s004
                                 + 0.0115449515987962 w_k1_MAT_s005
                                 + 0.0155664776089297 w_k1_MAT_s006
                                 + 0.043296810507029 w_k1_MAT_s007
                                 + 0.0178805315259688 w_k1_MAT_s010
                                 + 0.0218276691111228 w_k1_MAT_s011
                                 + 0.0310934646684067 w_k1_MAT_s012
                                 + 0.00736133887756817 w_k1_MAT_s013
                                 + 0.0239793353070573 w_k1_MAT_s014
                                 + 0.01868286368484 w_k1_MAT_s015
                                 + 0.00739138138211055 w_k1_MAT_s016
                                 + 0.0297515103816275 w_k1_MAT_s017
                                 + 0.0190720242675219 w_k1_MAT_s100
                                 + 0.0359843533185405 w_k1_MAT_s101
                                 + 0.0123841195075596 w_k1_MAT_s102
                                 + 0.0192285817761262 w_k1_MAT_s103
                                 + 0.0127408371702411 w_k1_MAT_s104
                                 + 0.0145457840008809 w_k1_MAT_s105
                                 + 0.0116691290649493 w_k1_MAT_s106
                                 + 0.0221606870712335 w_k1_MAT_s107
                                 + 0.0402721678967125 w_k1_MAT_s110
                                 + 0.0412532492922226 w_k1_MAT_s111
                                 + 0.0966535729382681 w_k1_MAT_s112
                                 + 0.106604538978318 w_k1_MAT_s113
                                 + 0.0400109325658397 w_k1_MAT_s114
                                 + 0.0642119924442502 w_k1_MAT_s115
                                 + 0.0430029992012676 w_k1_MAT_s116
                                 + 0.104366642670199 w_k1_MAT_s117
                                 + 0.0225783410107924 w_k1_MSFT_s000
                                 + 0.0306725210856429 w_k1_MSFT_s001
                                 + 0.0133090542990824 w_k1_MSFT_s002
                                 + 0.0392321942817861 w_k1_MSFT_s003
                                 + 0.0299530972334411 w_k1_MSFT_s004
                                 + 0.0132862903576097 w_k1_MSFT_s005
                                 + 0.0163833032961295 w_k1_MSFT_s006
                                 + 0.0515397703019922 w_k1_MSFT_s007
                                 + 0.0194966305135273 w_k1_MSFT_s010
                                 + 0.0166432615391402 w_k1_MSFT_s011
                                 + 0.0247971087228351 w_k1_MSFT_s012
                                 + 0.00690243981472281 w_k1_MSFT_s013
                                 + 0.0254885973979745 w_k1_MSFT_s014
                                 + 0.0168934576254471 w_k1_MSFT_s015
                                 + 0.00708591926757334 w_k1_MSFT_s016
                                 + 0.0240123016155237 w_k1_MSFT_s017
                                 + 0.0205469114544375 w_k1_MSFT_s100
                                 + 0.0331399364271675 w_k1_MSFT_s101
                                 + 0.0120457924330579 w_k1_MSFT_s102
                                 + 0.0226019708872245 w_k1_MSFT_s103
                                 + 0.0101144168365354 w_k1_MSFT_s104
                                 + 0.0123379589838628 w_k1_MSFT_s105
                                 + 0.0134768295564681 w_k1_MSFT_s106
                                 + 0.0200542637066633 w_k1_MSFT_s107
                                 + 0.0384794035532832 w_k1_MSFT_s110
                                 + 0.0409878936933232 w_k1_MSFT_s111
                                 + 0.0905926473292736 w_k1_MSFT_s112
                                 + 0.0703345529006071 w_k1_MSFT_s113
                                 + 0.0374142277736538 w_k1_MSFT_s114
                                 + 0.0620718428251823 w_k1_MSFT_s115
                                 + 0.0430655414589944 w_k1_MSFT_s116
                                 + 0.1088189670912 w_k1_MSFT_s117
                                 + 0.0212054154770533 w_k1_PG_s000
                                 + 0.0356215509569717 w_k1_PG_s001
                                 + 0.0138159152613979 w_k1_PG_s002
                                 + 0.0424357823963789 w_k1_PG_s003
                                 + 0.0295344191273755 w_k1_PG_s004
                                 + 0.0124143584525124 w_k1_PG_s005
                                 + 0.0165457234673031 w_k1_PG_s006
                                 + 0.0495482671540793 w_k1_PG_s007
                                 + 0.0174064423555713 w_k1_PG_s010
                                 + 0.0177424267058115 w_k1_PG_s011
                                 + 0.0265258766196515 w_k1_PG_s012
                                 + 0.0077448142923944 w_k1_PG_s013
                                 + 0.0277188502578729 w_k1_PG_s014
                                 + 0.0173546332486881 w_k1_PG_s015
                                 + 0.00676956864830229 w_k1_PG_s016
                                 + 0.0235648489574611 w_k1_PG_s017
                                 + 0.0192820488415003 w_k1_PG_s100
                                 + 0.0359943445980901 w_k1_PG_s101
                                 + 0.0121437933979277 w_k1_PG_s102
                                 + 0.0228689415870088 w_k1_PG_s103
                                 + 0.0108327330238418 w_k1_PG_s104
                                 + 0.0120605458959905 w_k1_PG_s105
                                 + 0.014450644996351 w_k1_PG_s106
                                 + 0.0199476389548826 w_k1_PG_s107
                                 + 0.0368539072572546 w_k1_PG_s110
                                 + 0.0417191508889497 w_k1_PG_s111
                                 + 0.0888497105720695 w_k1_PG_s112
                                 + 0.0680256550136681 w_k1_PG_s113
                                 + 0.0420980208740045 w_k1_PG_s114
                                 + 0.0628966632828062 w_k1_PG_s115
                                 + 0.042286364329581 w_k1_PG_s116
                                 + 0.111505806809247 w_k1_PG_s117 >= 1.17
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
