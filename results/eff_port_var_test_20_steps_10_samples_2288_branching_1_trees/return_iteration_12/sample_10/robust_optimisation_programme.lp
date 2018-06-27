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
 w_k1_ADM_s0_return_constraint:  1.15467 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.915685 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.16334 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.81314 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.09157 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.902705 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.893317 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.143 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.628408 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.836136 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.893232 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.822906 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.05064 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.12379 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.896814 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.01506 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.5456 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.96942 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.914429 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.27904 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.34029 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.759993 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.92563 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.33022 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.80676 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.3597 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.27529 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.05397 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.931296 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.885484 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.03755 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.891688 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.12787 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.756146 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.17129 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.84425 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.14862 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.783909 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23748 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.863067 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.20794 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.866374 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.19776 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.817779 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.803522 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.25414 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.59047 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.743044 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.841481 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.811909 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.55819 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.1781 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.623283 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.2111 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.55861 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.08024 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.822062 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.990653 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.05846 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.66345 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.845083 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.18473 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.813764 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.55484 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.2149 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.07192 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.06911 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.817513 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.02534 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.857799 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.39098 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.675502 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.54212 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.791117 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.18686 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.991894 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28336 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.8482 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.30732 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.790751 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.2769 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.800938 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.06785 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.01049 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.530338 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.775355 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.905303 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.72614 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.49262 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.18493 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.577038 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.0775 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.73865 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.899734 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.08489 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.43502 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.02653 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.607235 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.673216 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.07709 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.796377 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.81958 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.13438 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.21737 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.02302 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.812635 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.08701 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.09363 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.30117 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.577198 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.40268 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.642679 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.02902 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.01001 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1714 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.911144 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.16064 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.846434 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.1012 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.842088 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.976153 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.07583 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.708988 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.947376 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.83206 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.840057 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.26976 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.32792 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.776957 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.00705 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.42308 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.987973 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.912818 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.27177 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.21002 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.721939 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.805644 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.28627 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.838592 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.4606 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.12342 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.15381 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.931415 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.987933 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.0475 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.984908 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.11512 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.677267 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.30584 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.872902 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.11621 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.855793 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17456 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.916786 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.19386 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.884526 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.16524 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.824371 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.722903 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.902842 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.614958 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.592092 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.27216 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.848077 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.51479 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.69067 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.603949 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.64003 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.59427 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.93917 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.11291 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.87189 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.693553 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.06662 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.859114 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.66282 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.16057 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.25258 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.1709 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.91414 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.74941 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.619629 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.15392 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.860444 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 2.02997 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.84506 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.2934 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.574356 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.83274 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.53408 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.19528 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.968836 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.929667 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.8899 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.977056 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.821764 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.2042 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.29021 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.839529 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.908758 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.984811 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.894051 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.4232 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.13617 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.789576 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.0457 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.36252 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.953649 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.954771 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02445 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.969609 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.699705 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.02001 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.0243 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.999957 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.4352 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.16614 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.34221 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.07404 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.08028 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.09981 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.09735 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.07484 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.804214 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.32079 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.908418 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.05503 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.883367 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19165 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.902605 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.07855 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.905346 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.11432 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.876357 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.02585 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.982617 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.717596 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.773561 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.30463 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.791898 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.5313 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.929545 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.634632 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.29763 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.2809 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.89845 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.04117 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.016 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.808724 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.759557 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.897892 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.907594 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.05921 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.62903 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.20964 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.0676 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.36296 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.817927 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.12365 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.944147 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.22638 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.746489 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.24817 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.642992 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.996191 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.02997 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15666 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.8664 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.44977 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.857218 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.26251 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.894629 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.503381 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.795185 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.584341 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.566901 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.17476 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.762924 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.65738 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.689574 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.531143 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 2.04829 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.70501 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.25201 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.05708 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.820657 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.614695 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.925267 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.802956 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.49635 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.07834 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.40313 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.25826 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.718792 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.43732 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.730419 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.00286 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.728801 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.99131 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.64266 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.24214 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.589948 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.077 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.36421 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.20955 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.948795 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.03934 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.794019 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.06244 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.851881 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.14475 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.11017 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.7636 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.975781 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.968421 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.810549 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.20741 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.16201 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.888295 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.09405 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.42321 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.988424 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.1113 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.29802 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.01308 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.763873 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.8338 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.1418 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.921488 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.4516 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.08136 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.19179 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.00093 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.921622 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.01045 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.14141 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.01969 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.699506 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.25384 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.848801 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.0869 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.830021 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09692 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.943433 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.1086 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.943043 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.05311 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.872834 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.96005 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.0733 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.797823 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.930485 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.95766 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.933656 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.28284 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01676 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.765769 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04007 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.3204 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.944971 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.00343 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.05696 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.95956 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.805857 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.894916 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.05843 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.0115 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.31857 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.09903 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.10954 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.1833 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.935716 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.06884 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.02543 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.31649 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.793572 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.27913 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.864175 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01989 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.993486 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.255000783862703 z_k1_s0000
                                 - 0.0800000572665737 z_k1_s0001
                                 - 0.222999497275024 z_k1_s0002
                                 - 0.0950000949290786 z_k1_s0003
                                 - 0.216999884290356 z_k1_s0004
                                 - 0.321000834697629 z_k1_s0005
                                 - 0.169000108410626 z_k1_s0006
                                 - 0.0760000759432629 z_k1_s0007
                                 - 0.188000518753822 z_k1_s0010
                                 - 0.352000538252961 z_k1_s0011
                                 - 0.160000267224125 z_k1_s0012
                                 - 0.368999795672486 z_k1_s0013
                                 - 0.27000003210224 z_k1_s0014
                                 - 0.725001327203338 z_k1_s0015
                                 - 0.232000523985298 z_k1_s0016
                                 - 0.186000518516028 z_k1_s0017
                                 - 0.14899988470378 z_k1_s0020
                                 - 0.134999933530249 z_k1_s0021
                                 - 0.364999655952247 z_k1_s0022
                                 - 0.15400053390824 z_k1_s0023
                                 - 0.361000238188063 z_k1_s0024
                                 - 0.119999974915758 z_k1_s0025
                                 - 0.110999907947115 z_k1_s0026
                                 - 0.134999933530249 z_k1_s0027
                                 - 0.564999307217646 z_k1_s0030
                                 - 0.124999933561398 z_k1_s0031
                                 - 0.192999970562775 z_k1_s0032
                                 - 0.24300079138706 z_k1_s0033
                                 - 0.203999888004203 z_k1_s0034
                                 - 0.295998954241678 z_k1_s0035
                                 - 0.348999347095574 z_k1_s0036
                                 - 0.255000417505073 z_k1_s0037
                                 - 0.431001000903023 z_k1_s0040
                                 - 0.551003107690984 z_k1_s0041
                                 - 0.683003628603755 z_k1_s0042
                                 - 0.559000339437061 z_k1_s0043
                                 - 0.658000730121638 z_k1_s0044
                                 - 0.584999042128894 z_k1_s0045
                                 - 0.483000972792385 z_k1_s0046
                                 - 1.18300513512079 z_k1_s0047
                                 - 0.244999736157995 z_k1_s0050
                                 - 0.190999864123144 z_k1_s0051
                                 - 0.159000003302466 z_k1_s0052
                                 - 0.386000779848445 z_k1_s0053
                                 - 0.152999922498631 z_k1_s0054
                                 - 0.152999922498631 z_k1_s0055
                                 - 0.203000025730813 z_k1_s0056
                                 - 0.647999747217703 z_k1_s0057
                                 - 0.304999624208408 z_k1_s0060
                                 - 0.389999560397591 z_k1_s0061
                                 - 0.727000780534218 z_k1_s0062
                                 - 0.888000919056966 z_k1_s0063
                                 - 0.472001387966202 z_k1_s0064
                                 - 0.449000773892194 z_k1_s0065
                                 - 0.600998682562137 z_k1_s0066
                                 - 0.327999822282886 z_k1_s0067
                                 - 0.374000404284831 z_k1_s0070
                                 - 0.51900018302549 z_k1_s0071
                                 - 0.308000748741314 z_k1_s0072
                                 - 0.149000293365622 z_k1_s0073
                                 - 0.565000799751542 z_k1_s0074
                                 - 0.629999925012982 z_k1_s0075
                                 - 0.278000584011775 z_k1_s0076
                                 - 0.390001177582012 z_k1_s0077
                                 - 0.206999356867779 z_k1_s0100
                                 - 0.169000024949934 z_k1_s0101
                                 - 0.167000040270051 z_k1_s0102
                                 - 0.204000322347803 z_k1_s0103
                                 - 0.181999925369176 z_k1_s0104
                                 - 0.208999341547662 z_k1_s0105
                                 - 0.454999342173016 z_k1_s0106
                                 - 0.291999648262672 z_k1_s0107
                                 - 0.208000324909397 z_k1_s0110
                                 - 0.338000567202807 z_k1_s0111
                                 - 0.904002148329982 z_k1_s0112
                                 - 0.609001634718048 z_k1_s0113
                                 - 0.217000117195658 z_k1_s0114
                                 - 0.269000276873079 z_k1_s0115
                                 - 0.314999006024883 z_k1_s0116
                                 - 0.278000382959628 z_k1_s0117
                                 - 0.209999987832359 z_k1_s0120
                                 - 0.0629999963497078 z_k1_s0121
                                 - 0.0779999585377372 z_k1_s0122
                                 - 0.189999865848338 z_k1_s0123
                                 - 0.19400040744509 z_k1_s0124
                                 - 0.281999392574942 z_k1_s0125
                                 - 0.184999835352332 z_k1_s0126
                                 - 0.0910000378273514 z_k1_s0127
                                 - 0.542000399517506 z_k1_s0130
                                 - 0.639999131016414 z_k1_s0131
                                 - 0.327999922896068 z_k1_s0132
                                 - 0.430999982512857 z_k1_s0133
                                 - 0.867999987566505 z_k1_s0134
                                 - 0.48800019665038 z_k1_s0135
                                 - 0.537998747822952 z_k1_s0136
                                 - 1.07500076522383 z_k1_s0137
                                 - 0.583999313506191 z_k1_s0140
                                 - 0.544997311155198 z_k1_s0141
                                 - 0.15499942907564 z_k1_s0142
                                 - 0.232999361790431 z_k1_s0143
                                 - 0.273998928359714 z_k1_s0144
                                 - 0.182999431024457 z_k1_s0145
                                 - 0.192999468077492 z_k1_s0146
                                 - 0.378000026309034 z_k1_s0147
                                 - 0.123999855214929 z_k1_s0150
                                 - 0.203999850792257 z_k1_s0151
                                 - 0.503000744069553 z_k1_s0152
                                 - 0.141999923184792 z_k1_s0153
                                 - 0.421000340118001 z_k1_s0154
                                 - 0.11700001740433 z_k1_s0155
                                 - 0.252000102778523 z_k1_s0156
                                 - 0.358999775910863 z_k1_s0157
                                 - 0.538999900591996 z_k1_s0160
                                 - 0.158999638591316 z_k1_s0161
                                 - 0.426998718874214 z_k1_s0162
                                 - 0.205999467061847 z_k1_s0163
                                 - 0.661000024412127 z_k1_s0164
                                 - 0.651998919124518 z_k1_s0165
                                 - 0.377998416184715 z_k1_s0166
                                 - 0.406997405677024 z_k1_s0167
                                 - 0.101999938515047 z_k1_s0170
                                 - 0.248999276578152 z_k1_s0171
                                 - 0.345999156510135 z_k1_s0172
                                 - 0.285000278335971 z_k1_s0173
                                 - 0.131999967813571 z_k1_s0174
                                 - 0.100999991238407 z_k1_s0175
                                 - 0.150999932669325 z_k1_s0176
                                 - 0.244999165271728 z_k1_s0177
                                 - 0.547999043129226 z_k1_s1000
                                 - 0.32599967798823 z_k1_s1001
                                 - 0.886996829075893 z_k1_s1002
                                 - 0.561998970974196 z_k1_s1003
                                 - 0.770997257160652 z_k1_s1004
                                 - 0.332999344811105 z_k1_s1005
                                 - 1.57999949649372 z_k1_s1006
                                 - 0.934996242144806 z_k1_s1007
                                 - 0.31399965684705 z_k1_s1010
                                 - 0.165999774201357 z_k1_s1011
                                 - 0.438998770261528 z_k1_s1012
                                 - 0.11699977571127 z_k1_s1013
                                 - 0.307999372595522 z_k1_s1014
                                 - 0.127999736139701 z_k1_s1015
                                 - 0.370998752740614 z_k1_s1016
                                 - 0.559999296173186 z_k1_s1017
                                 - 0.576998919834043 z_k1_s1020
                                 - 0.627998463425122 z_k1_s1021
                                 - 0.412998581329953 z_k1_s1022
                                 - 0.193999716646777 z_k1_s1023
                                 - 0.588997870093104 z_k1_s1024
                                 - 0.393998908423084 z_k1_s1025
                                 - 0.819997687530377 z_k1_s1026
                                 - 0.389999124837061 z_k1_s1027
                                 - 0.276998872815619 z_k1_s1030
                                 - 0.825997842510697 z_k1_s1031
                                 - 0.36599867729908 z_k1_s1032
                                 - 0.243999245365581 z_k1_s1033
                                 - 0.510998277317381 z_k1_s1034
                                 - 0.417996552695759 z_k1_s1035
                                 - 0.850997444670246 z_k1_s1036
                                 - 0.321998919699974 z_k1_s1037
                                 - 0.955996775986694 z_k1_s1040
                                 - 0.456001174901236 z_k1_s1041
                                 - 0.364999230588288 z_k1_s1042
                                 - 0.217999749230463 z_k1_s1043
                                 - 1.0519972449129 z_k1_s1044
                                 - 0.240999782639998 z_k1_s1045
                                 - 0.481997881282822 z_k1_s1046
                                 - 0.4399989917504 z_k1_s1047
                                 - 0.473999455736942 z_k1_s1050
                                 - 0.274999617748292 z_k1_s1051
                                 - 0.170999934208768 z_k1_s1052
                                 - 0.406998632068266 z_k1_s1053
                                 - 0.554998655547119 z_k1_s1054
                                 - 0.528998806287168 z_k1_s1055
                                 - 0.114999840149286 z_k1_s1056
                                 - 0.339000789386646 z_k1_s1057
                                 - 0.530999119723665 z_k1_s1060
                                 - 0.418999936265138 z_k1_s1061
                                 - 0.722000718233945 z_k1_s1062
                                 - 0.522999178048056 z_k1_s1063
                                 - 0.363000743336556 z_k1_s1064
                                 - 0.261000290570495 z_k1_s1065
                                 - 0.342000697037742 z_k1_s1066
                                 - 0.827000949728018 z_k1_s1067
                                 - 0.71500245887399 z_k1_s1070
                                 - 0.36900045252268 z_k1_s1071
                                 - 1.45600394576015 z_k1_s1072
                                 - 0.389000439656108 z_k1_s1073
                                 - 0.697001090632047 z_k1_s1074
                                 - 0.692001889899762 z_k1_s1075
                                 - 0.477000701463621 z_k1_s1076
                                 - 0.512000944297478 z_k1_s1077
                                 - 0.373999620160038 z_k1_s1100
                                 - 0.691000475048318 z_k1_s1101
                                 - 0.25599972150208 z_k1_s1102
                                 - 0.658999212512077 z_k1_s1103
                                 - 0.464001008795747 z_k1_s1104
                                 - 0.67899952109779 z_k1_s1105
                                 - 0.464001008795747 z_k1_s1106
                                 - 0.25699954473159 z_k1_s1107
                                 - 0.631000187491707 z_k1_s1110
                                 - 0.365998904207124 z_k1_s1111
                                 - 0.187999693097402 z_k1_s1112
                                 - 0.394999300332361 z_k1_s1113
                                 - 0.354998550780505 z_k1_s1114
                                 - 0.466999471126834 z_k1_s1115
                                 - 0.231999633805293 z_k1_s1116
                                 - 0.31199877611127 z_k1_s1117
                                 - 0.206000645393376 z_k1_s1120
                                 - 0.16199949002726 z_k1_s1121
                                 - 0.111000043844813 z_k1_s1122
                                 - 0.263000261662119 z_k1_s1123
                                 - 0.125000072979307 z_k1_s1124
                                 - 0.0550000652069125 z_k1_s1125
                                 - 0.245000165103453 z_k1_s1126
                                 - 0.212000677579598 z_k1_s1127
                                 - 0.484000593429015 z_k1_s1130
                                 - 0.328999921897177 z_k1_s1131
                                 - 0.251000006819706 z_k1_s1132
                                 - 0.229999906441203 z_k1_s1133
                                 - 0.397000968246345 z_k1_s1134
                                 - 0.637999193338846 z_k1_s1135
                                 - 0.153000023886507 z_k1_s1136
                                 - 0.430998578422028 z_k1_s1137
                                 - 0.366002129902737 z_k1_s1140
                                 - 0.562001782026551 z_k1_s1141
                                 - 0.245000689708113 z_k1_s1142
                                 - 0.183000565149882 z_k1_s1143
                                 - 0.312001078242763 z_k1_s1144
                                 - 0.241000778407869 z_k1_s1145
                                 - 0.346000074394087 z_k1_s1146
                                 - 0.244000836833424 z_k1_s1147
                                 - 0.211000328942139 z_k1_s1150
                                 - 0.372000215309826 z_k1_s1151
                                 - 0.498000783076716 z_k1_s1152
                                 - 0.425999447208575 z_k1_s1153
                                 - 0.151000317053488 z_k1_s1154
                                 - 0.377001061968339 z_k1_s1155
                                 - 0.107000182468309 z_k1_s1156
                                 - 0.218000334262487 z_k1_s1157
                                 - 0.615002193482382 z_k1_s1160
                                 - 0.425000243254615 z_k1_s1161
                                 - 0.504000460092154 z_k1_s1162
                                 - 1.32100224430797 z_k1_s1163
                                 - 0.975000620690032 z_k1_s1164
                                 - 0.295000337962707 z_k1_s1165
                                 - 0.53700022718614 z_k1_s1166
                                 - 0.651998689543578 z_k1_s1167
                                 - 0.189000379467647 z_k1_s1170
                                 - 0.237000419745058 z_k1_s1171
                                 - 0.310000014707899 z_k1_s1172
                                 - 0.689999842736643 z_k1_s1173
                                 - 0.621001330965257 z_k1_s1174
                                 - 0.33300050029988 z_k1_s1175
                                 - 0.318001592089898 z_k1_s1176
                                 - 0.247000477219595 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.863024 w_k1_ADM_s000
                                 + 0.756323 w_k1_AXP_s000
                                 + 0.844533 w_k1_DIS_s000
                                 + 0.837299 w_k1_FDX_s000
                                 + 1.12994 w_k1_IBM_s000
                                 + 0.996818 w_k1_INTC_s000
                                 + 0.966516 w_k1_KO_s000
                                 + 0.739505 w_k1_MAT_s000
                                 + 0.855635 w_k1_MSFT_s000
                                 + 0.89306 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.949601 w_k1_ADM_s000
                                 + 0.65642 w_k1_AXP_s000
                                 + 0.600948 w_k1_DIS_s000
                                 + 0.723518 w_k1_FDX_s000
                                 + 0.750337 w_k1_IBM_s000
                                 + 0.704928 w_k1_INTC_s000
                                 + 0.659499 w_k1_KO_s000
                                 + 0.598693 w_k1_MAT_s000
                                 + 0.674646 w_k1_MSFT_s000
                                 + 0.74802 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.36207 w_k1_ADM_s000
                                 + 1.07737 w_k1_AXP_s000
                                 + 1.24715 w_k1_DIS_s000
                                 + 1.17238 w_k1_FDX_s000
                                 + 0.999772 w_k1_IBM_s000
                                 + 1.07726 w_k1_INTC_s000
                                 + 0.944701 w_k1_KO_s000
                                 + 0.883971 w_k1_MAT_s000
                                 + 1.16934 w_k1_MSFT_s000
                                 + 1.09704 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.33297 w_k1_ADM_s000
                                 + 1.49392 w_k1_AXP_s000
                                 + 1.28919 w_k1_DIS_s000
                                 + 1.34193 w_k1_FDX_s000
                                 + 0.919012 w_k1_IBM_s000
                                 + 1.07881 w_k1_INTC_s000
                                 + 0.836415 w_k1_KO_s000
                                 + 1.48088 w_k1_MAT_s000
                                 + 1.08874 w_k1_MSFT_s000
                                 + 1.11454 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.778761 w_k1_ADM_s000
                                 + 0.880751 w_k1_AXP_s000
                                 + 0.890946 w_k1_DIS_s000
                                 + 1.09124 w_k1_FDX_s000
                                 + 0.663956 w_k1_IBM_s000
                                 + 0.876293 w_k1_INTC_s000
                                 + 0.772414 w_k1_KO_s000
                                 + 0.774357 w_k1_MAT_s000
                                 + 0.877802 w_k1_MSFT_s000
                                 + 0.927579 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.932346 w_k1_ADM_s000
                                 + 0.867217 w_k1_AXP_s000
                                 + 0.798752 w_k1_DIS_s000
                                 + 0.900595 w_k1_FDX_s000
                                 + 0.81602 w_k1_IBM_s000
                                 + 0.838753 w_k1_INTC_s000
                                 + 0.818324 w_k1_KO_s000
                                 + 0.847866 w_k1_MAT_s000
                                 + 0.889678 w_k1_MSFT_s000
                                 + 0.921085 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.998591 w_k1_ADM_s000
                                 + 1.05787 w_k1_AXP_s000
                                 + 1.13035 w_k1_DIS_s000
                                 + 1.06369 w_k1_FDX_s000
                                 + 1.44824 w_k1_IBM_s000
                                 + 1.11843 w_k1_INTC_s000
                                 + 1.29174 w_k1_KO_s000 + 1.75334 w_k1_MAT_s000
                                 + 1.1491 w_k1_MSFT_s000 + 1.05918 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.46877 w_k1_ADM_s000
                                 + 1.25347 w_k1_AXP_s000
                                 + 1.55489 w_k1_DIS_s000
                                 + 1.25942 w_k1_FDX_s000
                                 + 1.63353 w_k1_IBM_s000
                                 + 1.29575 w_k1_INTC_s000
                                 + 1.55493 w_k1_KO_s000 + 1.85666 w_k1_MAT_s000
                                 + 1.29893 w_k1_MSFT_s000
                                 + 1.29471 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.02229 w_k1_ADM_s001
                                 + 0.913308 w_k1_AXP_s001
                                 + 1.03814 w_k1_DIS_s001
                                 + 1.09555 w_k1_FDX_s001
                                 + 0.730118 w_k1_IBM_s001
                                 + 1.00809 w_k1_INTC_s001
                                 + 0.835179 w_k1_KO_s001
                                 + 0.864876 w_k1_MAT_s001
                                 + 0.936546 w_k1_MSFT_s001
                                 + 0.921463 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.808382 w_k1_ADM_s001
                                 + 0.742547 w_k1_AXP_s001
                                 + 0.815769 w_k1_DIS_s001
                                 + 0.879598 w_k1_FDX_s001
                                 + 0.935568 w_k1_IBM_s001
                                 + 0.796805 w_k1_INTC_s001
                                 + 0.815816 w_k1_KO_s001
                                 + 0.73919 w_k1_MAT_s001
                                 + 0.786525 w_k1_MSFT_s001
                                 + 0.868139 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.944017 w_k1_ADM_s001
                                 + 1.19477 w_k1_AXP_s001
                                 + 0.895194 w_k1_DIS_s001
                                 + 0.927696 w_k1_FDX_s001
                                 + 0.887311 w_k1_IBM_s001
                                 + 0.802825 w_k1_INTC_s001
                                 + 0.974452 w_k1_KO_s001
                                 + 1.40414 w_k1_MAT_s001
                                 + 0.8242 w_k1_MSFT_s001
                                 + 0.885634 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.25049 w_k1_ADM_s001
                                 + 1.13346 w_k1_AXP_s001
                                 + 1.33544 w_k1_DIS_s001
                                 + 1.16974 w_k1_FDX_s001
                                 + 1.48017 w_k1_IBM_s001
                                 + 0.901617 w_k1_INTC_s001
                                 + 1.59787 w_k1_KO_s001 + 1.59945 w_k1_MAT_s001
                                 + 1.16047 w_k1_MSFT_s001
                                 + 1.26854 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.08571 w_k1_ADM_s001
                                 + 1.36113 w_k1_AXP_s001
                                 + 1.51398 w_k1_DIS_s001
                                 + 1.34875 w_k1_FDX_s001
                                 + 1.14646 w_k1_IBM_s001
                                 + 1.22304 w_k1_INTC_s001
                                 + 1.40723 w_k1_KO_s001 + 1.61085 w_k1_MAT_s001
                                 + 1.19047 w_k1_MSFT_s001
                                 + 1.26512 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.852082 w_k1_ADM_s001
                                 + 0.986829 w_k1_AXP_s001
                                 + 1.07399 w_k1_DIS_s001
                                 + 1.01542 w_k1_FDX_s001
                                 + 1.12332 w_k1_IBM_s001
                                 + 0.992773 w_k1_INTC_s001
                                 + 1.24711 w_k1_KO_s001 + 1.45424 w_k1_MAT_s001
                                 + 1.08891 w_k1_MSFT_s001
                                 + 0.985392 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.670401 w_k1_ADM_s001
                                 + 0.519398 w_k1_AXP_s001
                                 + 0.638194 w_k1_DIS_s001
                                 + 0.76425 w_k1_FDX_s001
                                 + 0.700323 w_k1_IBM_s001
                                 + 0.824743 w_k1_INTC_s001
                                 + 0.801849 w_k1_KO_s001
                                 + 0.67724 w_k1_MAT_s001
                                 + 0.779233 w_k1_MSFT_s001
                                 + 0.746006 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.945759 w_k1_ADM_s001
                                 + 0.980432 w_k1_AXP_s001
                                 + 0.745116 w_k1_DIS_s001
                                 + 0.950562 w_k1_FDX_s001
                                 + 0.815761 w_k1_IBM_s001
                                 + 0.774862 w_k1_INTC_s001
                                 + 0.800303 w_k1_KO_s001
                                 + 0.986887 w_k1_MAT_s001
                                 + 0.743746 w_k1_MSFT_s001
                                 + 0.82282 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.29418 w_k1_ADM_s002
                                 + 1.06907 w_k1_AXP_s002
                                 + 1.06763 w_k1_DIS_s002
                                 + 1.13246 w_k1_FDX_s002
                                 + 0.835226 w_k1_IBM_s002
                                 + 0.997939 w_k1_INTC_s002
                                 + 1.06221 w_k1_KO_s002 + 0.80804 w_k1_MAT_s002
                                 + 0.970062 w_k1_MSFT_s002
                                 + 0.955498 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.16083 w_k1_ADM_s002
                                 + 0.964303 w_k1_AXP_s002
                                 + 1.06597 w_k1_DIS_s002
                                 + 0.906718 w_k1_FDX_s002
                                 + 1.13263 w_k1_IBM_s002
                                 + 0.836333 w_k1_INTC_s002
                                 + 1.00708 w_k1_KO_s002 + 1.02742 w_k1_MAT_s002
                                 + 0.928371 w_k1_MSFT_s002
                                 + 1.00222 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.30399 w_k1_ADM_s002
                                 + 1.07347 w_k1_AXP_s002
                                 + 1.30494 w_k1_DIS_s002
                                 + 1.17581 w_k1_FDX_s002
                                 + 1.13909 w_k1_IBM_s002
                                 + 1.26337 w_k1_INTC_s002 + 1.2353 w_k1_KO_s002
                                 + 0.674609 w_k1_MAT_s002
                                 + 1.1607 w_k1_MSFT_s002 + 1.14546 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.16015 w_k1_ADM_s002
                                 + 0.787254 w_k1_AXP_s002
                                 + 1.08455 w_k1_DIS_s002
                                 + 1.04162 w_k1_FDX_s002
                                 + 0.739786 w_k1_IBM_s002
                                 + 1.06009 w_k1_INTC_s002
                                 + 0.93599 w_k1_KO_s002
                                 + 0.657361 w_k1_MAT_s002
                                 + 1.11589 w_k1_MSFT_s002
                                 + 0.974529 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.91859 w_k1_ADM_s002
                                 + 0.793534 w_k1_AXP_s002
                                 + 0.731289 w_k1_DIS_s002
                                 + 0.929459 w_k1_FDX_s002
                                 + 0.739708 w_k1_IBM_s002
                                 + 0.873789 w_k1_INTC_s002
                                 + 0.751138 w_k1_KO_s002
                                 + 0.55709 w_k1_MAT_s002
                                 + 0.873327 w_k1_MSFT_s002
                                 + 0.829919 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.983601 w_k1_ADM_s002
                                 + 0.896114 w_k1_AXP_s002
                                 + 0.732175 w_k1_DIS_s002
                                 + 0.764849 w_k1_FDX_s002
                                 + 1.22798 w_k1_IBM_s002
                                 + 1.05677 w_k1_INTC_s002
                                 + 0.956412 w_k1_KO_s002
                                 + 1.06674 w_k1_MAT_s002
                                 + 0.854027 w_k1_MSFT_s002
                                 + 0.907187 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.60967 w_k1_ADM_s002
                                 + 1.46176 w_k1_AXP_s002
                                 + 1.57647 w_k1_DIS_s002
                                 + 1.41831 w_k1_FDX_s002
                                 + 1.19403 w_k1_IBM_s002
                                 + 1.37771 w_k1_INTC_s002
                                 + 1.36238 w_k1_KO_s002 + 1.07903 w_k1_MAT_s002
                                 + 1.32742 w_k1_MSFT_s002
                                 + 1.47021 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.19631 w_k1_ADM_s002
                                 + 0.889629 w_k1_AXP_s002
                                 + 1.0448 w_k1_DIS_s002
                                 + 0.783199 w_k1_FDX_s002
                                 + 1.59223 w_k1_IBM_s002
                                 + 1.09446 w_k1_INTC_s002 + 1.2886 w_k1_KO_s002
                                 + 0.885091 w_k1_MAT_s002
                                 + 0.932731 w_k1_MSFT_s002
                                 + 0.96596 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.16043 w_k1_ADM_s003
                                 + 0.869664 w_k1_AXP_s003
                                 + 0.826895 w_k1_DIS_s003
                                 + 0.913435 w_k1_FDX_s003
                                 + 0.69868 w_k1_IBM_s003
                                 + 0.858686 w_k1_INTC_s003
                                 + 0.941474 w_k1_KO_s003
                                 + 0.740916 w_k1_MAT_s003
                                 + 0.975447 w_k1_MSFT_s003
                                 + 0.881753 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.57521 w_k1_ADM_s003
                                 + 1.46097 w_k1_AXP_s003
                                 + 1.53728 w_k1_DIS_s003
                                 + 1.50073 w_k1_FDX_s003
                                 + 1.04966 w_k1_IBM_s003
                                 + 1.26188 w_k1_INTC_s003
                                 + 1.19023 w_k1_KO_s003
                                 + 0.837408 w_k1_MAT_s003
                                 + 1.26196 w_k1_MSFT_s003
                                 + 1.03565 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.3426 w_k1_ADM_s003
                                 + 1.03567 w_k1_AXP_s003
                                 + 1.07198 w_k1_DIS_s003
                                 + 1.18003 w_k1_FDX_s003
                                 + 0.795904 w_k1_IBM_s003
                                 + 1.20353 w_k1_INTC_s003
                                 + 0.961572 w_k1_KO_s003
                                 + 0.548838 w_k1_MAT_s003
                                 + 1.01034 w_k1_MSFT_s003
                                 + 0.925109 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.1997 w_k1_ADM_s003
                                 + 1.17499 w_k1_AXP_s003
                                 + 0.906839 w_k1_DIS_s003
                                 + 0.967898 w_k1_FDX_s003
                                 + 1.00803 w_k1_IBM_s003
                                 + 0.924482 w_k1_INTC_s003
                                 + 0.952198 w_k1_KO_s003
                                 + 1.06603 w_k1_MAT_s003
                                 + 0.89556 w_k1_MSFT_s003
                                 + 0.954534 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.50035 w_k1_ADM_s003
                                 + 1.43116 w_k1_AXP_s003
                                 + 1.44106 w_k1_DIS_s003
                                 + 1.25414 w_k1_FDX_s003
                                 + 1.29607 w_k1_IBM_s003
                                 + 1.46093 w_k1_INTC_s003
                                 + 1.34957 w_k1_KO_s003 + 1.05199 w_k1_MAT_s003
                                 + 1.06471 w_k1_MSFT_s003
                                 + 1.13827 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.20819 w_k1_ADM_s003
                                 + 0.943246 w_k1_AXP_s003
                                 + 1.18352 w_k1_DIS_s003
                                 + 1.01227 w_k1_FDX_s003
                                 + 1.48866 w_k1_IBM_s003
                                 + 1.08209 w_k1_INTC_s003
                                 + 1.36783 w_k1_KO_s003 + 1.07881 w_k1_MAT_s003
                                 + 1.13697 w_k1_MSFT_s003
                                 + 1.10873 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.851971 w_k1_ADM_s003
                                 + 0.751432 w_k1_AXP_s003
                                 + 0.774752 w_k1_DIS_s003
                                 + 0.770265 w_k1_FDX_s003
                                 + 1.06028 w_k1_IBM_s003
                                 + 1.02785 w_k1_INTC_s003
                                 + 0.907547 w_k1_KO_s003
                                 + 0.565406 w_k1_MAT_s003
                                 + 0.764106 w_k1_MSFT_s003
                                 + 0.803238 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.36344 w_k1_ADM_s003
                                 + 1.33018 w_k1_AXP_s003
                                 + 1.11928 w_k1_DIS_s003
                                 + 1.13256 w_k1_FDX_s003
                                 + 1.21205 w_k1_IBM_s003
                                 + 1.32358 w_k1_INTC_s003
                                 + 1.13976 w_k1_KO_s003 + 1.06169 w_k1_MAT_s003
                                 + 1.10127 w_k1_MSFT_s003
                                 + 1.15496 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.860072 w_k1_ADM_s004
                                 + 1.10111 w_k1_AXP_s004
                                 + 0.829902 w_k1_DIS_s004
                                 + 1.0836 w_k1_FDX_s004
                                 + 0.684648 w_k1_IBM_s004
                                 + 1.14269 w_k1_INTC_s004
                                 + 0.670218 w_k1_KO_s004
                                 + 0.719713 w_k1_MAT_s004
                                 + 0.990069 w_k1_MSFT_s004
                                 + 0.923857 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.01493 w_k1_ADM_s004
                                 + 1.06339 w_k1_AXP_s004
                                 + 1.21308 w_k1_DIS_s004
                                 + 1.25854 w_k1_FDX_s004
                                 + 0.781846 w_k1_IBM_s004
                                 + 1.14278 w_k1_INTC_s004
                                 + 0.838669 w_k1_KO_s004
                                 + 0.734714 w_k1_MAT_s004
                                 + 1.16644 w_k1_MSFT_s004
                                 + 1.09281 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.829018 w_k1_ADM_s004
                                 + 1.04158 w_k1_AXP_s004
                                 + 0.80563 w_k1_DIS_s004
                                 + 0.871864 w_k1_FDX_s004
                                 + 1.08857 w_k1_IBM_s004
                                 + 0.902779 w_k1_INTC_s004
                                 + 0.83955 w_k1_KO_s004 + 1.01104 w_k1_MAT_s004
                                 + 0.772728 w_k1_MSFT_s004
                                 + 1.06192 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.15513 w_k1_ADM_s004
                                 + 1.46792 w_k1_AXP_s004
                                 + 1.10276 w_k1_DIS_s004
                                 + 1.14133 w_k1_FDX_s004
                                 + 0.927293 w_k1_IBM_s004
                                 + 1.25165 w_k1_INTC_s004
                                 + 0.795507 w_k1_KO_s004
                                 + 0.828496 w_k1_MAT_s004
                                 + 1.00024 w_k1_MSFT_s004
                                 + 1.12997 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.06157 w_k1_ADM_s004
                                 + 1.11294 w_k1_AXP_s004
                                 + 1.05822 w_k1_DIS_s004
                                 + 0.928483 w_k1_FDX_s004
                                 + 1.39687 w_k1_IBM_s004
                                 + 1.02871 w_k1_INTC_s004
                                 + 1.05088 w_k1_KO_s004 + 1.04945 w_k1_MAT_s004
                                 + 0.886511 w_k1_MSFT_s004
                                 + 1.00261 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.751114 w_k1_ADM_s004
                                 + 0.889071 w_k1_AXP_s004
                                 + 0.597192 w_k1_DIS_s004
                                 + 0.709806 w_k1_FDX_s004
                                 + 0.860344 w_k1_IBM_s004
                                 + 0.922464 w_k1_INTC_s004
                                 + 0.677538 w_k1_KO_s004
                                 + 0.701914 w_k1_MAT_s004
                                 + 0.696985 w_k1_MSFT_s004
                                 + 0.816421 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.861266 w_k1_ADM_s004
                                 + 0.692875 w_k1_AXP_s004
                                 + 0.796803 w_k1_DIS_s004
                                 + 0.851591 w_k1_FDX_s004
                                 + 0.683598 w_k1_IBM_s004
                                 + 0.901786 w_k1_INTC_s004
                                 + 0.722636 w_k1_KO_s004
                                 + 0.423972 w_k1_MAT_s004
                                 + 0.818147 w_k1_MSFT_s004
                                 + 0.862284 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.23284 w_k1_ADM_s004
                                 + 1.49812 w_k1_AXP_s004
                                 + 1.36836 w_k1_DIS_s004
                                 + 1.32458 w_k1_FDX_s004
                                 + 1.28622 w_k1_IBM_s004
                                 + 1.14272 w_k1_INTC_s004 + 1.264 w_k1_KO_s004
                                 + 1.73482 w_k1_MAT_s004
                                 + 1.10957 w_k1_MSFT_s004
                                 + 1.22778 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.819901 w_k1_ADM_s005
                                 + 1.06209 w_k1_AXP_s005
                                 + 1.04674 w_k1_DIS_s005
                                 + 1.18883 w_k1_FDX_s005
                                 + 0.896502 w_k1_IBM_s005
                                 + 1.27151 w_k1_INTC_s005
                                 + 1.20569 w_k1_KO_s005 + 0.8387 w_k1_MAT_s005
                                 + 1.26138 w_k1_MSFT_s005
                                 + 0.926176 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.919305 w_k1_ADM_s005
                                 + 0.842766 w_k1_AXP_s005
                                 + 0.965434 w_k1_DIS_s005
                                 + 0.998455 w_k1_FDX_s005
                                 + 0.903793 w_k1_IBM_s005
                                 + 1.04893 w_k1_INTC_s005
                                 + 1.20012 w_k1_KO_s005 + 1.02956 w_k1_MAT_s005
                                 + 1.05031 w_k1_MSFT_s005
                                 + 0.932378 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.675964 w_k1_ADM_s005
                                 + 0.659348 w_k1_AXP_s005
                                 + 0.719786 w_k1_DIS_s005
                                 + 0.755963 w_k1_FDX_s005
                                 + 0.626501 w_k1_IBM_s005
                                 + 0.909121 w_k1_INTC_s005
                                 + 0.747976 w_k1_KO_s005
                                 + 0.42418 w_k1_MAT_s005
                                 + 0.814669 w_k1_MSFT_s005
                                 + 0.732193 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.17781 w_k1_ADM_s005
                                 + 1.25332 w_k1_AXP_s005
                                 + 1.35551 w_k1_DIS_s005
                                 + 1.21148 w_k1_FDX_s005
                                 + 1.35608 w_k1_IBM_s005
                                 + 1.17517 w_k1_INTC_s005
                                 + 1.49582 w_k1_KO_s005 + 1.44601 w_k1_MAT_s005
                                 + 1.14906 w_k1_MSFT_s005
                                 + 1.14001 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.953816 w_k1_ADM_s005
                                 + 1.31803 w_k1_AXP_s005
                                 + 1.46333 w_k1_DIS_s005
                                 + 1.27039 w_k1_FDX_s005
                                 + 1.04215 w_k1_IBM_s005
                                 + 1.36259 w_k1_INTC_s005
                                 + 1.44034 w_k1_KO_s005 + 1.0762 w_k1_MAT_s005
                                 + 1.27515 w_k1_MSFT_s005 + 1.1296 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.924614 w_k1_ADM_s005
                                 + 0.881957 w_k1_AXP_s005
                                 + 0.817855 w_k1_DIS_s005
                                 + 0.96515 w_k1_FDX_s005
                                 + 0.669233 w_k1_IBM_s005
                                 + 0.933738 w_k1_INTC_s005
                                 + 0.82353 w_k1_KO_s005
                                 + 0.736111 w_k1_MAT_s005
                                 + 0.930774 w_k1_MSFT_s005
                                 + 0.903046 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.09108 w_k1_ADM_s005
                                 + 1.11002 w_k1_AXP_s005
                                 + 0.767585 w_k1_DIS_s005
                                 + 0.913605 w_k1_FDX_s005
                                 + 0.810097 w_k1_IBM_s005
                                 + 0.845829 w_k1_INTC_s005
                                 + 0.840316 w_k1_KO_s005
                                 + 1.25271 w_k1_MAT_s005
                                 + 0.784098 w_k1_MSFT_s005
                                 + 0.917513 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.653738 w_k1_ADM_s005
                                 + 0.845117 w_k1_AXP_s005
                                 + 0.831759 w_k1_DIS_s005
                                 + 0.834691 w_k1_FDX_s005
                                 + 1.11292 w_k1_IBM_s005
                                 + 1.08091 w_k1_INTC_s005
                                 + 1.08754 w_k1_KO_s005
                                 + 0.925345 w_k1_MAT_s005
                                 + 0.98957 w_k1_MSFT_s005
                                 + 0.941141 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.911588 w_k1_ADM_s006
                                 + 0.551124 w_k1_AXP_s006
                                 + 0.781904 w_k1_DIS_s006
                                 + 0.723753 w_k1_FDX_s006
                                 + 1.07713 w_k1_IBM_s006
                                 + 0.870526 w_k1_INTC_s006
                                 + 0.951405 w_k1_KO_s006
                                 + 0.693871 w_k1_MAT_s006
                                 + 0.974359 w_k1_MSFT_s006
                                 + 0.923431 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.873462 w_k1_ADM_s006
                                 + 0.849713 w_k1_AXP_s006
                                 + 0.884652 w_k1_DIS_s006
                                 + 0.769929 w_k1_FDX_s006
                                 + 1.57021 w_k1_IBM_s006
                                 + 0.807811 w_k1_INTC_s006
                                 + 1.10472 w_k1_KO_s006 + 1.16932 w_k1_MAT_s006
                                 + 0.814485 w_k1_MSFT_s006
                                 + 0.878761 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.887559 w_k1_ADM_s006
                                 + 0.833485 w_k1_AXP_s006
                                 + 0.978115 w_k1_DIS_s006
                                 + 0.962137 w_k1_FDX_s006
                                 + 0.834348 w_k1_IBM_s006
                                 + 0.97328 w_k1_INTC_s006
                                 + 0.869263 w_k1_KO_s006
                                 + 0.74025 w_k1_MAT_s006
                                 + 1.04418 w_k1_MSFT_s006
                                 + 0.989745 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.17399 w_k1_ADM_s006
                                 + 0.751802 w_k1_AXP_s006
                                 + 0.965095 w_k1_DIS_s006
                                 + 1.00846 w_k1_FDX_s006
                                 + 0.641876 w_k1_IBM_s006
                                 + 1.12537 w_k1_INTC_s006
                                 + 0.765661 w_k1_KO_s006
                                 + 0.537484 w_k1_MAT_s006
                                 + 1.1045 w_k1_MSFT_s006
                                 + 0.879235 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.68059 w_k1_ADM_s006
                                 + 1.67077 w_k1_AXP_s006
                                 + 1.68079 w_k1_DIS_s006
                                 + 1.38061 w_k1_FDX_s006
                                 + 1.48854 w_k1_IBM_s006
                                 + 1.22389 w_k1_INTC_s006
                                 + 1.24354 w_k1_KO_s006 + 1.31677 w_k1_MAT_s006
                                 + 1.35146 w_k1_MSFT_s006
                                 + 1.25875 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.992178 w_k1_ADM_s006
                                 + 0.844845 w_k1_AXP_s006
                                 + 0.737281 w_k1_DIS_s006
                                 + 0.816335 w_k1_FDX_s006
                                 + 1.08401 w_k1_IBM_s006
                                 + 0.810505 w_k1_INTC_s006
                                 + 0.779401 w_k1_KO_s006
                                 + 1.03248 w_k1_MAT_s006
                                 + 0.876436 w_k1_MSFT_s006
                                 + 0.824757 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.40868 w_k1_ADM_s006
                                 + 1.05888 w_k1_AXP_s006
                                 + 1.52556 w_k1_DIS_s006
                                 + 1.18781 w_k1_FDX_s006
                                 + 1.03101 w_k1_IBM_s006
                                 + 1.23668 w_k1_INTC_s006
                                 + 1.22935 w_k1_KO_s006
                                 + 0.934181 w_k1_MAT_s006
                                 + 1.30373 w_k1_MSFT_s006
                                 + 1.03334 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.03688 w_k1_ADM_s006
                                 + 1.08788 w_k1_AXP_s006
                                 + 0.814299 w_k1_DIS_s006
                                 + 1.02496 w_k1_FDX_s006
                                 + 0.841852 w_k1_IBM_s006
                                 + 0.902469 w_k1_INTC_s006
                                 + 0.867663 w_k1_KO_s006
                                 + 1.10445 w_k1_MAT_s006
                                 + 0.864796 w_k1_MSFT_s006
                                 + 0.895669 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.09662 w_k1_ADM_s007
                                 + 1.33731 w_k1_AXP_s007
                                 + 1.03545 w_k1_DIS_s007
                                 + 1.15509 w_k1_FDX_s007
                                 + 1.13952 w_k1_IBM_s007
                                 + 1.36779 w_k1_INTC_s007
                                 + 1.12928 w_k1_KO_s007 + 1.15876 w_k1_MAT_s007
                                 + 1.09093 w_k1_MSFT_s007 + 1.1521 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.05568 w_k1_ADM_s007
                                 + 0.92819 w_k1_AXP_s007
                                 + 1.11479 w_k1_DIS_s007
                                 + 0.977949 w_k1_FDX_s007
                                 + 0.927874 w_k1_IBM_s007
                                 + 1.07395 w_k1_INTC_s007
                                 + 1.14984 w_k1_KO_s007 + 1.01711 w_k1_MAT_s007
                                 + 1.09598 w_k1_MSFT_s007
                                 + 1.08661 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.09899 w_k1_ADM_s007
                                 + 0.930023 w_k1_AXP_s007
                                 + 0.737449 w_k1_DIS_s007
                                 + 0.842329 w_k1_FDX_s007
                                 + 0.862924 w_k1_IBM_s007
                                 + 0.939353 w_k1_INTC_s007
                                 + 0.858081 w_k1_KO_s007
                                 + 0.758346 w_k1_MAT_s007
                                 + 0.975513 w_k1_MSFT_s007
                                 + 0.99213 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.896704 w_k1_ADM_s007
                                 + 0.646055 w_k1_AXP_s007
                                 + 0.798567 w_k1_DIS_s007
                                 + 0.793469 w_k1_FDX_s007
                                 + 0.7346 w_k1_IBM_s007
                                 + 0.868007 w_k1_INTC_s007
                                 + 0.959797 w_k1_KO_s007
                                 + 0.728987 w_k1_MAT_s007
                                 + 0.915948 w_k1_MSFT_s007
                                 + 0.867406 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.931662 w_k1_ADM_s007
                                 + 0.826589 w_k1_AXP_s007
                                 + 0.923844 w_k1_DIS_s007
                                 + 0.800978 w_k1_FDX_s007
                                 + 1.3421 w_k1_IBM_s007
                                 + 1.13895 w_k1_INTC_s007
                                 + 1.31529 w_k1_KO_s007 + 0.94479 w_k1_MAT_s007
                                 + 1.04457 w_k1_MSFT_s007
                                 + 0.948113 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.32945 w_k1_ADM_s007
                                 + 1.40411 w_k1_AXP_s007
                                 + 1.53914 w_k1_DIS_s007
                                 + 1.22346 w_k1_FDX_s007
                                 + 1.31916 w_k1_IBM_s007
                                 + 1.61511 w_k1_INTC_s007
                                 + 1.29889 w_k1_KO_s007 + 1.60809 w_k1_MAT_s007
                                 + 1.32252 w_k1_MSFT_s007
                                 + 1.14271 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.75367 w_k1_ADM_s007
                                 + 0.715977 w_k1_AXP_s007
                                 + 0.557688 w_k1_DIS_s007
                                 + 0.692095 w_k1_FDX_s007
                                 + 0.7409 w_k1_IBM_s007
                                 + 0.907072 w_k1_INTC_s007
                                 + 0.697055 w_k1_KO_s007
                                 + 0.647633 w_k1_MAT_s007
                                 + 0.819338 w_k1_MSFT_s007
                                 + 0.856927 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.992229 w_k1_ADM_s007
                                 + 0.954465 w_k1_AXP_s007
                                 + 0.984479 w_k1_DIS_s007
                                 + 1.01413 w_k1_FDX_s007
                                 + 0.804572 w_k1_IBM_s007
                                 + 1.20757 w_k1_INTC_s007
                                 + 1.00989 w_k1_KO_s007
                                 + 0.617451 w_k1_MAT_s007
                                 + 1.04293 w_k1_MSFT_s007
                                 + 0.985608 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.808187 w_k1_ADM_s010
                                 + 1.19914 w_k1_AXP_s010
                                 + 1.50295 w_k1_DIS_s010 + 1.1119 w_k1_FDX_s010
                                 + 1.25209 w_k1_IBM_s010
                                 + 1.03314 w_k1_INTC_s010
                                 + 1.19525 w_k1_KO_s010 + 1.51471 w_k1_MAT_s010
                                 + 1.03322 w_k1_MSFT_s010
                                 + 1.10478 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.3035 w_k1_ADM_s010
                                 + 1.69415 w_k1_AXP_s010
                                 + 1.76264 w_k1_DIS_s010
                                 + 1.32652 w_k1_FDX_s010
                                 + 1.51155 w_k1_IBM_s010
                                 + 1.06154 w_k1_INTC_s010
                                 + 1.46139 w_k1_KO_s010 + 1.86685 w_k1_MAT_s010
                                 + 1.2092 w_k1_MSFT_s010 + 1.3576 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.40145 w_k1_ADM_s010
                                 + 1.20346 w_k1_AXP_s010
                                 + 1.14815 w_k1_DIS_s010
                                 + 1.25894 w_k1_FDX_s010
                                 + 0.709471 w_k1_IBM_s010
                                 + 1.06027 w_k1_INTC_s010
                                 + 1.04489 w_k1_KO_s010 + 0.9125 w_k1_MAT_s010
                                 + 1.15434 w_k1_MSFT_s010
                                 + 1.17815 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.914486 w_k1_ADM_s010
                                 + 1.01045 w_k1_AXP_s010
                                 + 1.31588 w_k1_DIS_s010 + 1.1897 w_k1_FDX_s010
                                 + 0.963767 w_k1_IBM_s010
                                 + 1.09944 w_k1_INTC_s010
                                 + 1.01294 w_k1_KO_s010 + 1.02689 w_k1_MAT_s010
                                 + 1.18429 w_k1_MSFT_s010
                                 + 1.07306 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.928699 w_k1_ADM_s010
                                 + 1.10485 w_k1_AXP_s010
                                 + 1.12847 w_k1_DIS_s010
                                 + 0.992909 w_k1_FDX_s010
                                 + 1.60727 w_k1_IBM_s010
                                 + 0.960618 w_k1_INTC_s010
                                 + 1.24284 w_k1_KO_s010 + 1.48723 w_k1_MAT_s010
                                 + 0.914434 w_k1_MSFT_s010
                                 + 1.09865 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.757445 w_k1_ADM_s010
                                 + 0.7116 w_k1_AXP_s010
                                 + 0.645885 w_k1_DIS_s010
                                 + 0.776192 w_k1_FDX_s010
                                 + 0.645321 w_k1_IBM_s010
                                 + 0.841403 w_k1_INTC_s010
                                 + 0.663779 w_k1_KO_s010
                                 + 0.576298 w_k1_MAT_s010
                                 + 0.79161 w_k1_MSFT_s010
                                 + 0.778988 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.94737 w_k1_ADM_s010
                                 + 0.960428 w_k1_AXP_s010
                                 + 0.822944 w_k1_DIS_s010
                                 + 0.862609 w_k1_FDX_s010
                                 + 1.05034 w_k1_IBM_s010
                                 + 0.86301 w_k1_INTC_s010
                                 + 0.849893 w_k1_KO_s010 + 1.2557 w_k1_MAT_s010
                                 + 0.792766 w_k1_MSFT_s010
                                 + 0.985806 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.848254 w_k1_ADM_s010
                                 + 0.73535 w_k1_AXP_s010
                                 + 1.03842 w_k1_DIS_s010
                                 + 0.944815 w_k1_FDX_s010
                                 + 0.80222 w_k1_IBM_s010
                                 + 1.01201 w_k1_INTC_s010
                                 + 0.955102 w_k1_KO_s010
                                 + 0.751265 w_k1_MAT_s010
                                 + 0.949713 w_k1_MSFT_s010
                                 + 1.00791 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.935118 w_k1_ADM_s011
                                 + 0.704827 w_k1_AXP_s011
                                 + 0.796538 w_k1_DIS_s011
                                 + 0.921071 w_k1_FDX_s011
                                 + 0.593726 w_k1_IBM_s011
                                 + 0.781813 w_k1_INTC_s011
                                 + 0.740365 w_k1_KO_s011
                                 + 0.51376 w_k1_MAT_s011
                                 + 0.909745 w_k1_MSFT_s011
                                 + 1.03669 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.25174 w_k1_ADM_s011
                                 + 0.91693 w_k1_AXP_s011
                                 + 0.939177 w_k1_DIS_s011
                                 + 0.963584 w_k1_FDX_s011
                                 + 1.18166 w_k1_IBM_s011
                                 + 0.79325 w_k1_INTC_s011
                                 + 0.879917 w_k1_KO_s011
                                 + 0.976638 w_k1_MAT_s011
                                 + 0.860729 w_k1_MSFT_s011
                                 + 0.980259 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.03454 w_k1_ADM_s011
                                 + 0.709707 w_k1_AXP_s011
                                 + 0.798193 w_k1_DIS_s011
                                 + 0.823992 w_k1_FDX_s011
                                 + 0.920211 w_k1_IBM_s011
                                 + 0.744737 w_k1_INTC_s011
                                 + 0.761081 w_k1_KO_s011
                                 + 0.820215 w_k1_MAT_s011
                                 + 0.848736 w_k1_MSFT_s011
                                 + 0.918239 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.26819 w_k1_ADM_s011
                                 + 1.1178 w_k1_AXP_s011 + 1.27898 w_k1_DIS_s011
                                 + 1.1965 w_k1_FDX_s011
                                 + 0.740836 w_k1_IBM_s011
                                 + 0.99743 w_k1_INTC_s011
                                 + 0.788432 w_k1_KO_s011
                                 + 0.736215 w_k1_MAT_s011
                                 + 1.09966 w_k1_MSFT_s011
                                 + 0.987801 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.885151 w_k1_ADM_s011
                                 + 0.976161 w_k1_AXP_s011
                                 + 1.51008 w_k1_DIS_s011
                                 + 1.12556 w_k1_FDX_s011 + 1.2366 w_k1_IBM_s011
                                 + 1.05957 w_k1_INTC_s011
                                 + 1.14052 w_k1_KO_s011 + 1.07782 w_k1_MAT_s011
                                 + 1.14865 w_k1_MSFT_s011
                                 + 1.12315 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.06459 w_k1_ADM_s011
                                 + 0.872116 w_k1_AXP_s011
                                 + 1.08809 w_k1_DIS_s011
                                 + 0.928192 w_k1_FDX_s011
                                 + 1.46555 w_k1_IBM_s011
                                 + 0.99589 w_k1_INTC_s011
                                 + 1.12385 w_k1_KO_s011 + 1.27297 w_k1_MAT_s011
                                 + 1.01374 w_k1_MSFT_s011
                                 + 1.08138 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.20976 w_k1_ADM_s011
                                 + 1.39181 w_k1_AXP_s011
                                 + 1.78911 w_k1_DIS_s011
                                 + 1.50048 w_k1_FDX_s011
                                 + 1.07476 w_k1_IBM_s011
                                 + 1.24694 w_k1_INTC_s011
                                 + 1.28068 w_k1_KO_s011
                                 + 0.973004 w_k1_MAT_s011
                                 + 1.25588 w_k1_MSFT_s011
                                 + 1.15191 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.54894 w_k1_ADM_s011
                                 + 1.41092 w_k1_AXP_s011
                                 + 1.49184 w_k1_DIS_s011
                                 + 1.20607 w_k1_FDX_s011
                                 + 1.66915 w_k1_IBM_s011
                                 + 1.26661 w_k1_INTC_s011
                                 + 1.57785 w_k1_KO_s011 + 1.68181 w_k1_MAT_s011
                                 + 1.26007 w_k1_MSFT_s011
                                 + 1.44137 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.04937 w_k1_ADM_s012
                                 + 1.18716 w_k1_AXP_s012
                                 + 1.15726 w_k1_DIS_s012
                                 + 1.04234 w_k1_FDX_s012
                                 + 1.71243 w_k1_IBM_s012
                                 + 0.772853 w_k1_INTC_s012
                                 + 1.45407 w_k1_KO_s012 + 1.93026 w_k1_MAT_s012
                                 + 0.931898 w_k1_MSFT_s012
                                 + 1.00842 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.02837 w_k1_ADM_s012
                                 + 1.09111 w_k1_AXP_s012 + 1.4232 w_k1_DIS_s012
                                 + 1.32221 w_k1_FDX_s012
                                 + 0.662572 w_k1_IBM_s012
                                 + 1.19019 w_k1_INTC_s012
                                 + 1.02594 w_k1_KO_s012
                                 + 0.815181 w_k1_MAT_s012
                                 + 1.23499 w_k1_MSFT_s012
                                 + 0.930715 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.791206 w_k1_ADM_s012
                                 + 0.958081 w_k1_AXP_s012
                                 + 0.913454 w_k1_DIS_s012
                                 + 1.02149 w_k1_FDX_s012
                                 + 0.894925 w_k1_IBM_s012
                                 + 0.837196 w_k1_INTC_s012
                                 + 0.983773 w_k1_KO_s012
                                 + 1.05871 w_k1_MAT_s012
                                 + 0.907007 w_k1_MSFT_s012
                                 + 0.842901 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.933414 w_k1_ADM_s012
                                 + 1.48433 w_k1_AXP_s012
                                 + 1.50262 w_k1_DIS_s012
                                 + 1.29061 w_k1_FDX_s012
                                 + 0.96185 w_k1_IBM_s012
                                 + 1.47277 w_k1_INTC_s012
                                 + 1.29655 w_k1_KO_s012 + 1.07396 w_k1_MAT_s012
                                 + 1.19478 w_k1_MSFT_s012
                                 + 1.17484 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.818602 w_k1_ADM_s012
                                 + 0.865554 w_k1_AXP_s012
                                 + 0.913348 w_k1_DIS_s012
                                 + 0.886133 w_k1_FDX_s012
                                 + 1.19461 w_k1_IBM_s012
                                 + 0.708533 w_k1_INTC_s012
                                 + 1.18589 w_k1_KO_s012 + 1.45398 w_k1_MAT_s012
                                 + 0.847933 w_k1_MSFT_s012
                                 + 0.895115 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.975896 w_k1_ADM_s012
                                 + 1.12684 w_k1_AXP_s012
                                 + 0.950974 w_k1_DIS_s012
                                 + 1.02084 w_k1_FDX_s012
                                 + 0.795577 w_k1_IBM_s012
                                 + 0.798466 w_k1_INTC_s012
                                 + 0.910256 w_k1_KO_s012
                                 + 0.954491 w_k1_MAT_s012
                                 + 0.906846 w_k1_MSFT_s012
                                 + 0.808779 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.768412 w_k1_ADM_s012
                                 + 0.776805 w_k1_AXP_s012
                                 + 0.609998 w_k1_DIS_s012
                                 + 0.793759 w_k1_FDX_s012
                                 + 0.718624 w_k1_IBM_s012
                                 + 0.751602 w_k1_INTC_s012
                                 + 0.8047 w_k1_KO_s012 + 0.712806 w_k1_MAT_s012
                                 + 0.831472 w_k1_MSFT_s012
                                 + 0.793708 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.866298 w_k1_ADM_s012
                                 + 0.843644 w_k1_AXP_s012
                                 + 1.25318 w_k1_DIS_s012
                                 + 1.02076 w_k1_FDX_s012
                                 + 0.82225 w_k1_IBM_s012
                                 + 0.955156 w_k1_INTC_s012
                                 + 1.16572 w_k1_KO_s012 + 0.84741 w_k1_MAT_s012
                                 + 1.00416 w_k1_MSFT_s012
                                 + 0.959677 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.978454 w_k1_ADM_s013
                                 + 0.650079 w_k1_AXP_s013
                                 + 0.79711 w_k1_DIS_s013
                                 + 0.844176 w_k1_FDX_s013
                                 + 0.908409 w_k1_IBM_s013
                                 + 0.994127 w_k1_INTC_s013
                                 + 0.85425 w_k1_KO_s013
                                 + 0.688338 w_k1_MAT_s013
                                 + 0.958365 w_k1_MSFT_s013
                                 + 0.92246 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.46709 w_k1_ADM_s013
                                 + 1.20801 w_k1_AXP_s013
                                 + 1.28661 w_k1_DIS_s013
                                 + 1.10554 w_k1_FDX_s013
                                 + 1.56499 w_k1_IBM_s013
                                 + 1.00165 w_k1_INTC_s013
                                 + 1.28293 w_k1_KO_s013 + 1.22642 w_k1_MAT_s013
                                 + 1.14183 w_k1_MSFT_s013
                                 + 1.10715 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.994089 w_k1_ADM_s013
                                 + 0.504605 w_k1_AXP_s013
                                 + 0.662759 w_k1_DIS_s013
                                 + 0.774132 w_k1_FDX_s013
                                 + 0.641012 w_k1_IBM_s013
                                 + 0.610248 w_k1_INTC_s013
                                 + 0.651681 w_k1_KO_s013
                                 + 0.521091 w_k1_MAT_s013
                                 + 0.765111 w_k1_MSFT_s013
                                 + 0.800084 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.12007 w_k1_ADM_s013
                                 + 0.896963 w_k1_AXP_s013
                                 + 0.961768 w_k1_DIS_s013
                                 + 1.08319 w_k1_FDX_s013
                                 + 0.708526 w_k1_IBM_s013
                                 + 0.917781 w_k1_INTC_s013
                                 + 0.808954 w_k1_KO_s013
                                 + 0.756554 w_k1_MAT_s013
                                 + 0.942963 w_k1_MSFT_s013
                                 + 0.972046 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.1863 w_k1_ADM_s013
                                 + 0.84814 w_k1_AXP_s013
                                 + 0.821661 w_k1_DIS_s013
                                 + 0.865879 w_k1_FDX_s013
                                 + 1.02848 w_k1_IBM_s013
                                 + 0.987726 w_k1_INTC_s013
                                 + 0.967526 w_k1_KO_s013
                                 + 0.822656 w_k1_MAT_s013
                                 + 0.898512 w_k1_MSFT_s013
                                 + 0.962515 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11466 w_k1_ADM_s013
                                 + 0.953761 w_k1_AXP_s013
                                 + 0.810081 w_k1_DIS_s013
                                 + 0.874238 w_k1_FDX_s013
                                 + 1.44058 w_k1_IBM_s013
                                 + 0.885387 w_k1_INTC_s013
                                 + 1.1365 w_k1_KO_s013 + 1.29422 w_k1_MAT_s013
                                 + 0.925298 w_k1_MSFT_s013
                                 + 1.02435 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.12975 w_k1_ADM_s013
                                 + 0.905228 w_k1_AXP_s013
                                 + 0.806037 w_k1_DIS_s013
                                 + 0.956051 w_k1_FDX_s013
                                 + 0.79739 w_k1_IBM_s013
                                 + 0.909143 w_k1_INTC_s013
                                 + 0.849121 w_k1_KO_s013
                                 + 0.727511 w_k1_MAT_s013
                                 + 0.946822 w_k1_MSFT_s013
                                 + 0.944988 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.37394 w_k1_ADM_s013
                                 + 1.19274 w_k1_AXP_s013
                                 + 1.34311 w_k1_DIS_s013
                                 + 1.31713 w_k1_FDX_s013
                                 + 1.04654 w_k1_IBM_s013
                                 + 1.25103 w_k1_INTC_s013
                                 + 1.13661 w_k1_KO_s013 + 1.05336 w_k1_MAT_s013
                                 + 1.32494 w_k1_MSFT_s013
                                 + 1.21265 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.894312 w_k1_ADM_s014
                                 + 0.689588 w_k1_AXP_s014
                                 + 0.691226 w_k1_DIS_s014
                                 + 0.86118 w_k1_FDX_s014
                                 + 0.753978 w_k1_IBM_s014
                                 + 0.74708 w_k1_INTC_s014
                                 + 0.797336 w_k1_KO_s014
                                 + 0.788643 w_k1_MAT_s014
                                 + 0.861107 w_k1_MSFT_s014
                                 + 0.839249 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.19795 w_k1_ADM_s014
                                 + 1.10927 w_k1_AXP_s014
                                 + 0.960514 w_k1_DIS_s014
                                 + 0.993481 w_k1_FDX_s014
                                 + 1.09201 w_k1_IBM_s014
                                 + 0.987204 w_k1_INTC_s014
                                 + 0.917271 w_k1_KO_s014
                                 + 1.07635 w_k1_MAT_s014
                                 + 0.915776 w_k1_MSFT_s014
                                 + 0.946398 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 2.10877 w_k1_ADM_s014
                                 + 1.60335 w_k1_AXP_s014
                                 + 1.69794 w_k1_DIS_s014
                                 + 1.48569 w_k1_FDX_s014
                                 + 1.17708 w_k1_IBM_s014 + 1.318 w_k1_INTC_s014
                                 + 1.34322 w_k1_KO_s014 + 1.29224 w_k1_MAT_s014
                                 + 1.25042 w_k1_MSFT_s014 + 1.3545 w_k1_PG_s014
                                 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20154 w_k1_ADM_s014
                                 + 1.25835 w_k1_AXP_s014
                                 + 1.34023 w_k1_DIS_s014
                                 + 1.21695 w_k1_FDX_s014
                                 + 1.18628 w_k1_IBM_s014
                                 + 1.24762 w_k1_INTC_s014
                                 + 1.09098 w_k1_KO_s014 + 1.11006 w_k1_MAT_s014
                                 + 1.07177 w_k1_MSFT_s014
                                 + 1.10939 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.807232 w_k1_ADM_s014
                                 + 0.689517 w_k1_AXP_s014
                                 + 0.712648 w_k1_DIS_s014
                                 + 0.773665 w_k1_FDX_s014
                                 + 1.19237 w_k1_IBM_s014
                                 + 0.880764 w_k1_INTC_s014
                                 + 0.84176 w_k1_KO_s014
                                 + 0.982545 w_k1_MAT_s014
                                 + 0.736124 w_k1_MSFT_s014
                                 + 0.826136 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.987842 w_k1_ADM_s014
                                 + 1.13612 w_k1_AXP_s014
                                 + 1.10406 w_k1_DIS_s014
                                 + 1.17869 w_k1_FDX_s014
                                 + 0.780119 w_k1_IBM_s014
                                 + 0.995923 w_k1_INTC_s014
                                 + 0.89299 w_k1_KO_s014
                                 + 0.879897 w_k1_MAT_s014
                                 + 1.06839 w_k1_MSFT_s014
                                 + 1.00546 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.24576 w_k1_ADM_s014
                                 + 1.07936 w_k1_AXP_s014
                                 + 0.835322 w_k1_DIS_s014
                                 + 1.11245 w_k1_FDX_s014
                                 + 0.866385 w_k1_IBM_s014
                                 + 0.98881 w_k1_INTC_s014
                                 + 0.811914 w_k1_KO_s014
                                 + 0.849839 w_k1_MAT_s014
                                 + 0.865445 w_k1_MSFT_s014
                                 + 0.930739 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.05854 w_k1_ADM_s014
                                 + 1.20789 w_k1_AXP_s014
                                 + 1.17813 w_k1_DIS_s014
                                 + 1.09823 w_k1_FDX_s014
                                 + 1.58193 w_k1_IBM_s014
                                 + 1.04993 w_k1_INTC_s014
                                 + 1.13307 w_k1_KO_s014 + 1.84295 w_k1_MAT_s014
                                 + 1.01093 w_k1_MSFT_s014 + 1.1407 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.368 w_k1_ADM_s015
                                 + 1.26132 w_k1_AXP_s015
                                 + 1.80027 w_k1_DIS_s015
                                 + 1.27481 w_k1_FDX_s015
                                 + 1.43864 w_k1_IBM_s015
                                 + 1.28436 w_k1_INTC_s015
                                 + 1.66524 w_k1_KO_s015 + 1.73553 w_k1_MAT_s015
                                 + 1.40812 w_k1_MSFT_s015
                                 + 1.41006 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11705 w_k1_ADM_s015
                                 + 1.60814 w_k1_AXP_s015
                                 + 1.45247 w_k1_DIS_s015
                                 + 1.39503 w_k1_FDX_s015
                                 + 0.995816 w_k1_IBM_s015
                                 + 1.26899 w_k1_INTC_s015
                                 + 1.12418 w_k1_KO_s015 + 1.60728 w_k1_MAT_s015
                                 + 1.21522 w_k1_MSFT_s015
                                 + 1.02138 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.916379 w_k1_ADM_s015
                                 + 1.05862 w_k1_AXP_s015
                                 + 0.896826 w_k1_DIS_s015
                                 + 0.937665 w_k1_FDX_s015
                                 + 1.17506 w_k1_IBM_s015
                                 + 0.912831 w_k1_INTC_s015
                                 + 1.05334 w_k1_KO_s015 + 1.06373 w_k1_MAT_s015
                                 + 0.835076 w_k1_MSFT_s015
                                 + 0.983975 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.890398 w_k1_ADM_s015
                                 + 1.35406 w_k1_AXP_s015
                                 + 1.32205 w_k1_DIS_s015
                                 + 1.02883 w_k1_FDX_s015
                                 + 1.57964 w_k1_IBM_s015
                                 + 1.29755 w_k1_INTC_s015
                                 + 1.41014 w_k1_KO_s015 + 1.71497 w_k1_MAT_s015
                                 + 1.04684 w_k1_MSFT_s015 + 1.17 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.630239 w_k1_ADM_s015
                                 + 0.661358 w_k1_AXP_s015
                                 + 0.682062 w_k1_DIS_s015
                                 + 0.72919 w_k1_FDX_s015
                                 + 0.808748 w_k1_IBM_s015
                                 + 0.940411 w_k1_INTC_s015
                                 + 0.868665 w_k1_KO_s015
                                 + 0.783108 w_k1_MAT_s015
                                 + 0.924931 w_k1_MSFT_s015
                                 + 0.857428 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.972127 w_k1_ADM_s015
                                 + 1.01453 w_k1_AXP_s015
                                 + 1.16014 w_k1_DIS_s015
                                 + 0.991673 w_k1_FDX_s015
                                 + 1.20772 w_k1_IBM_s015
                                 + 1.13956 w_k1_INTC_s015
                                 + 1.27901 w_k1_KO_s015 + 1.53172 w_k1_MAT_s015
                                 + 1.18568 w_k1_MSFT_s015
                                 + 1.07958 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.05876 w_k1_ADM_s015
                                 + 1.27116 w_k1_AXP_s015
                                 + 1.17102 w_k1_DIS_s015
                                 + 1.16144 w_k1_FDX_s015
                                 + 1.02534 w_k1_IBM_s015
                                 + 1.07148 w_k1_INTC_s015
                                 + 1.12738 w_k1_KO_s015 + 1.42258 w_k1_MAT_s015
                                 + 1.15085 w_k1_MSFT_s015
                                 + 1.09234 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.909368 w_k1_ADM_s015
                                 + 0.870393 w_k1_AXP_s015
                                 + 0.938478 w_k1_DIS_s015
                                 + 0.951522 w_k1_FDX_s015
                                 + 0.671542 w_k1_IBM_s015
                                 + 0.895922 w_k1_INTC_s015
                                 + 0.782576 w_k1_KO_s015
                                 + 0.805572 w_k1_MAT_s015
                                 + 0.953259 w_k1_MSFT_s015
                                 + 0.921337 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.03458 w_k1_ADM_s016
                                 + 1.3137 w_k1_AXP_s016 + 1.15296 w_k1_DIS_s016
                                 + 1.1099 w_k1_FDX_s016 + 1.32701 w_k1_IBM_s016
                                 + 1.23549 w_k1_INTC_s016
                                 + 1.21333 w_k1_KO_s016 + 1.69007 w_k1_MAT_s016
                                 + 1.1968 w_k1_MSFT_s016 + 1.04642 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.24691 w_k1_ADM_s016
                                 + 1.54146 w_k1_AXP_s016
                                 + 1.65375 w_k1_DIS_s016
                                 + 1.34107 w_k1_FDX_s016
                                 + 1.83659 w_k1_IBM_s016
                                 + 1.0717 w_k1_INTC_s016 + 1.56157 w_k1_KO_s016
                                 + 2.23624 w_k1_MAT_s016
                                 + 1.38736 w_k1_MSFT_s016
                                 + 1.40244 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.04075 w_k1_ADM_s016
                                 + 1.08622 w_k1_AXP_s016
                                 + 1.18813 w_k1_DIS_s016
                                 + 1.25462 w_k1_FDX_s016
                                 + 0.696501 w_k1_IBM_s016
                                 + 1.13187 w_k1_INTC_s016
                                 + 0.968724 w_k1_KO_s016
                                 + 1.13225 w_k1_MAT_s016
                                 + 1.3888 w_k1_MSFT_s016 + 1.06242 w_k1_PG_s016
                                 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.00864 w_k1_ADM_s016
                                 + 0.955356 w_k1_AXP_s016
                                 + 0.946833 w_k1_DIS_s016
                                 + 0.838028 w_k1_FDX_s016
                                 + 1.45329 w_k1_IBM_s016
                                 + 0.960233 w_k1_INTC_s016
                                 + 1.2336 w_k1_KO_s016 + 1.79067 w_k1_MAT_s016
                                 + 0.961649 w_k1_MSFT_s016
                                 + 1.05287 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.792726 w_k1_ADM_s016
                                 + 0.649519 w_k1_AXP_s016
                                 + 0.734499 w_k1_DIS_s016
                                 + 0.814504 w_k1_FDX_s016
                                 + 0.690141 w_k1_IBM_s016
                                 + 0.901735 w_k1_INTC_s016
                                 + 0.808421 w_k1_KO_s016
                                 + 0.630408 w_k1_MAT_s016
                                 + 0.869203 w_k1_MSFT_s016
                                 + 0.860403 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.869491 w_k1_ADM_s016
                                 + 0.763446 w_k1_AXP_s016
                                 + 0.769777 w_k1_DIS_s016
                                 + 0.788538 w_k1_FDX_s016
                                 + 1.01508 w_k1_IBM_s016
                                 + 0.881154 w_k1_INTC_s016
                                 + 1.02154 w_k1_KO_s016 + 1.1651 w_k1_MAT_s016
                                 + 0.873333 w_k1_MSFT_s016
                                 + 0.952693 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.15902 w_k1_ADM_s016
                                 + 1.59086 w_k1_AXP_s016
                                 + 1.54415 w_k1_DIS_s016
                                 + 1.36875 w_k1_FDX_s016
                                 + 1.17779 w_k1_IBM_s016
                                 + 1.28703 w_k1_INTC_s016
                                 + 1.25647 w_k1_KO_s016 + 1.81773 w_k1_MAT_s016
                                 + 1.5162 w_k1_MSFT_s016 + 1.16454 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.01494 w_k1_ADM_s016
                                 + 0.990024 w_k1_AXP_s016
                                 + 0.940603 w_k1_DIS_s016
                                 + 0.926808 w_k1_FDX_s016
                                 + 0.900837 w_k1_IBM_s016
                                 + 0.968219 w_k1_INTC_s016
                                 + 0.859292 w_k1_KO_s016
                                 + 1.28372 w_k1_MAT_s016
                                 + 1.04972 w_k1_MSFT_s016
                                 + 0.957104 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.957071 w_k1_ADM_s017
                                 + 0.950749 w_k1_AXP_s017
                                 + 1.08079 w_k1_DIS_s017
                                 + 0.847605 w_k1_FDX_s017
                                 + 1.44658 w_k1_IBM_s017
                                 + 1.19787 w_k1_INTC_s017
                                 + 1.23115 w_k1_KO_s017 + 1.3333 w_k1_MAT_s017
                                 + 0.958356 w_k1_MSFT_s017
                                 + 1.02438 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.31263 w_k1_ADM_s017
                                 + 0.975766 w_k1_AXP_s017
                                 + 1.38172 w_k1_DIS_s017
                                 + 1.12309 w_k1_FDX_s017
                                 + 0.913697 w_k1_IBM_s017
                                 + 1.068 w_k1_INTC_s017 + 1.17181 w_k1_KO_s017
                                 + 1.09507 w_k1_MAT_s017
                                 + 1.15414 w_k1_MSFT_s017
                                 + 1.06345 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.878582 w_k1_ADM_s017
                                 + 0.793617 w_k1_AXP_s017
                                 + 0.831585 w_k1_DIS_s017
                                 + 0.861848 w_k1_FDX_s017
                                 + 0.91679 w_k1_IBM_s017
                                 + 0.875704 w_k1_INTC_s017
                                 + 0.858494 w_k1_KO_s017
                                 + 0.812616 w_k1_MAT_s017
                                 + 0.919778 w_k1_MSFT_s017
                                 + 0.983694 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.05247 w_k1_ADM_s017
                                 + 0.922247 w_k1_AXP_s017
                                 + 0.875035 w_k1_DIS_s017
                                 + 0.933106 w_k1_FDX_s017
                                 + 0.624492 w_k1_IBM_s017
                                 + 1.10482 w_k1_INTC_s017
                                 + 0.868674 w_k1_KO_s017
                                 + 0.865096 w_k1_MAT_s017
                                 + 1.00862 w_k1_MSFT_s017
                                 + 0.862723 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.999982 w_k1_ADM_s017
                                 + 1.08641 w_k1_AXP_s017
                                 + 1.19098 w_k1_DIS_s017
                                 + 0.932176 w_k1_FDX_s017
                                 + 1.27651 w_k1_IBM_s017
                                 + 1.12659 w_k1_INTC_s017
                                 + 1.26094 w_k1_KO_s017 + 1.49914 w_k1_MAT_s017
                                 + 1.10001 w_k1_MSFT_s017
                                 + 1.04817 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.926264 w_k1_ADM_s017
                                 + 0.730703 w_k1_AXP_s017
                                 + 0.65462 w_k1_DIS_s017
                                 + 0.713686 w_k1_FDX_s017
                                 + 1.00172 w_k1_IBM_s017
                                 + 0.775509 w_k1_INTC_s017
                                 + 0.783341 w_k1_KO_s017
                                 + 0.997744 w_k1_MAT_s017
                                 + 0.82033 w_k1_MSFT_s017
                                 + 0.856446 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.21101 w_k1_ADM_s017
                                 + 1.67384 w_k1_AXP_s017
                                 + 1.37357 w_k1_DIS_s017
                                 + 1.18379 w_k1_FDX_s017 + 1.7511 w_k1_IBM_s017
                                 + 1.30827 w_k1_INTC_s017 + 1.4644 w_k1_KO_s017
                                 + 2.2927 w_k1_MAT_s017
                                 + 1.09109 w_k1_MSFT_s017
                                 + 1.19353 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.18368 w_k1_ADM_s017
                                 + 1.22491 w_k1_AXP_s017
                                 + 1.38375 w_k1_DIS_s017
                                 + 1.22531 w_k1_FDX_s017
                                 + 1.10471 w_k1_IBM_s017
                                 + 1.28146 w_k1_INTC_s017
                                 + 1.04369 w_k1_KO_s017 + 1.25947 w_k1_MAT_s017
                                 + 1.27041 w_k1_MSFT_s017
                                 + 1.11436 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.861651 w_k1_ADM_s100
                                 + 0.823594 w_k1_AXP_s100
                                 + 1.09836 w_k1_DIS_s100
                                 + 0.954776 w_k1_FDX_s100
                                 + 1.07961 w_k1_IBM_s100
                                 + 0.927514 w_k1_INTC_s100
                                 + 0.868567 w_k1_KO_s100
                                 + 1.03838 w_k1_MAT_s100
                                 + 1.04872 w_k1_MSFT_s100
                                 + 0.921417 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.783395 w_k1_ADM_s100
                                 + 0.795713 w_k1_AXP_s100
                                 + 1.21844 w_k1_DIS_s100
                                 + 0.952458 w_k1_FDX_s100
                                 + 1.13645 w_k1_IBM_s100
                                 + 1.08392 w_k1_INTC_s100
                                 + 1.03884 w_k1_KO_s100
                                 + 0.967875 w_k1_MAT_s100
                                 + 1.15403 w_k1_MSFT_s100
                                 + 1.05417 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.111 w_k1_ADM_s100
                                 + 1.30196 w_k1_AXP_s100
                                 + 1.33505 w_k1_DIS_s100
                                 + 1.24584 w_k1_FDX_s100
                                 + 1.26203 w_k1_IBM_s100
                                 + 1.19438 w_k1_INTC_s100
                                 + 1.24535 w_k1_KO_s100 + 1.26843 w_k1_MAT_s100
                                 + 1.22854 w_k1_MSFT_s100
                                 + 1.15109 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.792296 w_k1_ADM_s100
                                 + 0.942997 w_k1_AXP_s100
                                 + 0.854651 w_k1_DIS_s100
                                 + 0.872346 w_k1_FDX_s100
                                 + 1.32782 w_k1_IBM_s100
                                 + 0.751899 w_k1_INTC_s100
                                 + 0.977773 w_k1_KO_s100
                                 + 1.32042 w_k1_MAT_s100
                                 + 0.842426 w_k1_MSFT_s100
                                 + 0.893129 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.20415 w_k1_ADM_s100
                                 + 1.28495 w_k1_AXP_s100
                                 + 1.03566 w_k1_DIS_s100
                                 + 1.07495 w_k1_FDX_s100 + 1.0733 w_k1_IBM_s100
                                 + 0.996339 w_k1_INTC_s100
                                 + 0.853775 w_k1_KO_s100
                                 + 1.14272 w_k1_MAT_s100
                                 + 0.97184 w_k1_MSFT_s100
                                 + 1.06974 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.32277 w_k1_ADM_s100
                                 + 1.68532 w_k1_AXP_s100
                                 + 1.94002 w_k1_DIS_s100
                                 + 1.43914 w_k1_FDX_s100 + 1.3401 w_k1_IBM_s100
                                 + 1.29539 w_k1_INTC_s100
                                 + 1.45484 w_k1_KO_s100 + 1.31357 w_k1_MAT_s100
                                 + 1.4123 w_k1_MSFT_s100 + 1.44715 w_k1_PG_s100
                                 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.741493 w_k1_ADM_s100
                                 + 0.731655 w_k1_AXP_s100
                                 + 0.711588 w_k1_DIS_s100
                                 + 0.797603 w_k1_FDX_s100
                                 + 0.820373 w_k1_IBM_s100
                                 + 0.822114 w_k1_INTC_s100
                                 + 0.753362 w_k1_KO_s100
                                 + 0.847946 w_k1_MAT_s100
                                 + 0.937681 w_k1_MSFT_s100
                                 + 0.872207 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.02135 w_k1_ADM_s100
                                 + 0.965853 w_k1_AXP_s100
                                 + 1.08509 w_k1_DIS_s100
                                 + 1.07691 w_k1_FDX_s100
                                 + 0.771141 w_k1_IBM_s100
                                 + 0.942153 w_k1_INTC_s100
                                 + 0.835157 w_k1_KO_s100
                                 + 0.712051 w_k1_MAT_s100
                                 + 1.0011 w_k1_MSFT_s100
                                 + 0.995532 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.22169 w_k1_ADM_s101
                                 + 1.3087 w_k1_AXP_s101 + 1.60703 w_k1_DIS_s101
                                 + 1.33332 w_k1_FDX_s101
                                 + 1.13785 w_k1_IBM_s101
                                 + 1.47642 w_k1_INTC_s101
                                 + 1.29435 w_k1_KO_s101 + 1.06319 w_k1_MAT_s101
                                 + 1.35081 w_k1_MSFT_s101
                                 + 1.10952 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.977294 w_k1_ADM_s101
                                 + 1.39824 w_k1_AXP_s101 + 1.1404 w_k1_DIS_s101
                                 + 1.10284 w_k1_FDX_s101
                                 + 1.45771 w_k1_IBM_s101
                                 + 1.30683 w_k1_INTC_s101 + 1.3013 w_k1_KO_s101
                                 + 2.08298 w_k1_MAT_s101
                                 + 1.11466 w_k1_MSFT_s101
                                 + 1.07051 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.898049 w_k1_ADM_s101
                                 + 0.713251 w_k1_AXP_s101
                                 + 0.607861 w_k1_DIS_s101
                                 + 0.780813 w_k1_FDX_s101
                                 + 0.73301 w_k1_IBM_s101
                                 + 0.950863 w_k1_INTC_s101
                                 + 0.73023 w_k1_KO_s101
                                 + 0.990105 w_k1_MAT_s101
                                 + 0.784564 w_k1_MSFT_s101
                                 + 0.800837 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.0311 w_k1_ADM_s101
                                 + 0.977511 w_k1_AXP_s101
                                 + 1.26356 w_k1_DIS_s101 + 1.1041 w_k1_FDX_s101
                                 + 1.07482 w_k1_IBM_s101
                                 + 1.23747 w_k1_INTC_s101
                                 + 1.06793 w_k1_KO_s101 + 1.18811 w_k1_MAT_s101
                                 + 1.14609 w_k1_MSFT_s101
                                 + 1.10871 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.842363 w_k1_ADM_s101
                                 + 1.09734 w_k1_AXP_s101
                                 + 0.978882 w_k1_DIS_s101
                                 + 1.03874 w_k1_FDX_s101
                                 + 1.04209 w_k1_IBM_s101
                                 + 1.15106 w_k1_INTC_s101
                                 + 0.938182 w_k1_KO_s101
                                 + 1.56062 w_k1_MAT_s101
                                 + 1.00928 w_k1_MSFT_s101
                                 + 0.901802 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.03932 w_k1_ADM_s101
                                 + 1.06545 w_k1_AXP_s101
                                 + 1.24511 w_k1_DIS_s101
                                 + 0.964604 w_k1_FDX_s101
                                 + 1.52302 w_k1_IBM_s101
                                 + 1.34968 w_k1_INTC_s101
                                 + 1.42843 w_k1_KO_s101 + 2.30865 w_k1_MAT_s101
                                 + 1.06179 w_k1_MSFT_s101
                                 + 1.03236 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.09766 w_k1_ADM_s101
                                 + 0.898669 w_k1_AXP_s101
                                 + 0.949688 w_k1_DIS_s101
                                 + 1.08554 w_k1_FDX_s101
                                 + 0.80237 w_k1_IBM_s101
                                 + 1.34399 w_k1_INTC_s101
                                 + 0.960768 w_k1_KO_s101 + 1.1137 w_k1_MAT_s101
                                 + 1.08189 w_k1_MSFT_s101
                                 + 0.916483 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.83013 w_k1_ADM_s101
                                 + 0.935011 w_k1_AXP_s101
                                 + 0.837827 w_k1_DIS_s101
                                 + 0.84513 w_k1_FDX_s101
                                 + 1.17453 w_k1_IBM_s101
                                 + 1.12688 w_k1_INTC_s101
                                 + 0.830617 w_k1_KO_s101
                                 + 1.74503 w_k1_MAT_s101
                                 + 0.817906 w_k1_MSFT_s101
                                 + 0.911238 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.22083 w_k1_ADM_s102
                                 + 1.66084 w_k1_AXP_s102
                                 + 1.25185 w_k1_DIS_s102
                                 + 1.28881 w_k1_FDX_s102
                                 + 1.29469 w_k1_IBM_s102
                                 + 1.14987 w_k1_INTC_s102
                                 + 1.13514 w_k1_KO_s102 + 1.46161 w_k1_MAT_s102
                                 + 1.04497 w_k1_MSFT_s102 + 1.0475 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.19972 w_k1_ADM_s102
                                 + 1.10753 w_k1_AXP_s102
                                 + 1.21287 w_k1_DIS_s102
                                 + 1.24625 w_k1_FDX_s102
                                 + 0.884914 w_k1_IBM_s102
                                 + 1.17418 w_k1_INTC_s102
                                 + 0.979083 w_k1_KO_s102
                                 + 0.929251 w_k1_MAT_s102
                                 + 1.05326 w_k1_MSFT_s102
                                 + 0.935435 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.07978 w_k1_ADM_s102
                                 + 1.04419 w_k1_AXP_s102
                                 + 1.00581 w_k1_DIS_s102
                                 + 0.988954 w_k1_FDX_s102
                                 + 1.56769 w_k1_IBM_s102
                                 + 1.17717 w_k1_INTC_s102 + 1.117 w_k1_KO_s102
                                 + 1.64298 w_k1_MAT_s102
                                 + 0.971178 w_k1_MSFT_s102
                                 + 0.965448 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.949798 w_k1_ADM_s102
                                 + 0.883913 w_k1_AXP_s102
                                 + 0.694954 w_k1_DIS_s102
                                 + 0.87911 w_k1_FDX_s102
                                 + 0.570919 w_k1_IBM_s102
                                 + 0.861973 w_k1_INTC_s102
                                 + 0.636518 w_k1_KO_s102
                                 + 0.648652 w_k1_MAT_s102
                                 + 0.854687 w_k1_MSFT_s102
                                 + 0.77913 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.0274 w_k1_ADM_s102
                                 + 1.0364 w_k1_AXP_s102 + 1.03662 w_k1_DIS_s102
                                 + 0.991775 w_k1_FDX_s102
                                 + 0.997982 w_k1_IBM_s102
                                 + 1.02501 w_k1_INTC_s102
                                 + 0.959537 w_k1_KO_s102
                                 + 0.973265 w_k1_MAT_s102
                                 + 0.910205 w_k1_MSFT_s102
                                 + 0.948898 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.08956 w_k1_ADM_s102
                                 + 0.922774 w_k1_AXP_s102
                                 + 1.42533 w_k1_DIS_s102
                                 + 1.07692 w_k1_FDX_s102
                                 + 1.25811 w_k1_IBM_s102
                                 + 1.10965 w_k1_INTC_s102
                                 + 1.25302 w_k1_KO_s102
                                 + 0.979043 w_k1_MAT_s102
                                 + 1.15179 w_k1_MSFT_s102
                                 + 1.03579 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.968004 w_k1_ADM_s102
                                 + 0.870373 w_k1_AXP_s102
                                 + 0.765631 w_k1_DIS_s102
                                 + 0.855605 w_k1_FDX_s102
                                 + 0.831371 w_k1_IBM_s102
                                 + 0.835631 w_k1_INTC_s102
                                 + 0.806638 w_k1_KO_s102
                                 + 0.804942 w_k1_MAT_s102
                                 + 0.856756 w_k1_MSFT_s102
                                 + 0.888421 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.04762 w_k1_ADM_s102
                                 + 1.05144 w_k1_AXP_s102
                                 + 0.892612 w_k1_DIS_s102
                                 + 0.857405 w_k1_FDX_s102
                                 + 1.19789 w_k1_IBM_s102
                                 + 0.955485 w_k1_INTC_s102
                                 + 0.922192 w_k1_KO_s102
                                 + 1.09267 w_k1_MAT_s102
                                 + 0.899835 w_k1_MSFT_s102
                                 + 0.847137 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.04013 w_k1_ADM_s103
                                 + 0.684662 w_k1_AXP_s103
                                 + 0.582038 w_k1_DIS_s103
                                 + 0.784245 w_k1_FDX_s103
                                 + 0.564419 w_k1_IBM_s103
                                 + 0.650397 w_k1_INTC_s103
                                 + 0.58052 w_k1_KO_s103
                                 + 0.614691 w_k1_MAT_s103
                                 + 0.660675 w_k1_MSFT_s103
                                 + 0.735699 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.993203 w_k1_ADM_s103
                                 + 1.11027 w_k1_AXP_s103
                                 + 0.805732 w_k1_DIS_s103
                                 + 0.991932 w_k1_FDX_s103
                                 + 0.996408 w_k1_IBM_s103
                                 + 0.889417 w_k1_INTC_s103
                                 + 0.836108 w_k1_KO_s103
                                 + 1.15823 w_k1_MAT_s103
                                 + 0.949974 w_k1_MSFT_s103
                                 + 0.914898 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.20112 w_k1_ADM_s103
                                 + 1.50996 w_k1_AXP_s103
                                 + 1.63559 w_k1_DIS_s103
                                 + 1.52017 w_k1_FDX_s103
                                 + 1.05808 w_k1_IBM_s103
                                 + 1.1398 w_k1_INTC_s103 + 1.04104 w_k1_KO_s103
                                 + 1.0127 w_k1_MAT_s103
                                 + 1.25899 w_k1_MSFT_s103
                                 + 1.17413 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.78475 w_k1_ADM_s103
                                 + 1.31724 w_k1_AXP_s103
                                 + 1.36637 w_k1_DIS_s103
                                 + 1.06915 w_k1_FDX_s103
                                 + 1.97088 w_k1_IBM_s103
                                 + 1.22632 w_k1_INTC_s103 + 1.2228 w_k1_KO_s103
                                 + 1.6404 w_k1_MAT_s103 + 1.0054 w_k1_MSFT_s103
                                 + 0.990316 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.815655 w_k1_ADM_s103
                                 + 0.623527 w_k1_AXP_s103
                                 + 0.800335 w_k1_DIS_s103
                                 + 0.869767 w_k1_FDX_s103
                                 + 0.696616 w_k1_IBM_s103
                                 + 0.693807 w_k1_INTC_s103
                                 + 0.746137 w_k1_KO_s103
                                 + 0.633259 w_k1_MAT_s103
                                 + 0.875341 w_k1_MSFT_s103
                                 + 0.809852 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.16151 w_k1_ADM_s103
                                 + 0.796234 w_k1_AXP_s103
                                 + 1.13751 w_k1_DIS_s103
                                 + 1.20256 w_k1_FDX_s103
                                 + 0.800848 w_k1_IBM_s103
                                 + 1.05806 w_k1_INTC_s103
                                 + 0.853208 w_k1_KO_s103
                                 + 0.662468 w_k1_MAT_s103
                                 + 1.08078 w_k1_MSFT_s103
                                 + 0.976956 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.07767 w_k1_ADM_s103
                                 + 1.03326 w_k1_AXP_s103
                                 + 0.981952 w_k1_DIS_s103
                                 + 0.915126 w_k1_FDX_s103
                                 + 1.31762 w_k1_IBM_s103
                                 + 0.975032 w_k1_INTC_s103
                                 + 1.02954 w_k1_KO_s103
                                 + 0.895425 w_k1_MAT_s103
                                 + 0.929979 w_k1_MSFT_s103
                                 + 1.12905 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.09047 w_k1_ADM_s103
                                 + 1.27855 w_k1_AXP_s103
                                 + 0.898385 w_k1_DIS_s103
                                 + 1.19588 w_k1_FDX_s103
                                 + 0.885761 w_k1_IBM_s103
                                 + 1.31809 w_k1_INTC_s103
                                 + 0.868804 w_k1_KO_s103
                                 + 0.942892 w_k1_MAT_s103
                                 + 0.974931 w_k1_MSFT_s103
                                 + 1.10081 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.04539 w_k1_ADM_s104
                                 + 1.58546 w_k1_AXP_s104
                                 + 1.47141 w_k1_DIS_s104
                                 + 1.22012 w_k1_FDX_s104
                                 + 1.32028 w_k1_IBM_s104
                                 + 1.24727 w_k1_INTC_s104
                                 + 1.28122 w_k1_KO_s104 + 1.28544 w_k1_MAT_s104
                                 + 1.15873 w_k1_MSFT_s104
                                 + 1.16181 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.974834 w_k1_ADM_s104
                                 + 1.1141 w_k1_AXP_s104 + 1.03747 w_k1_DIS_s104
                                 + 1.06174 w_k1_FDX_s104
                                 + 0.894483 w_k1_IBM_s104
                                 + 0.928937 w_k1_INTC_s104
                                 + 0.96425 w_k1_KO_s104
                                 + 0.911702 w_k1_MAT_s104
                                 + 1.02491 w_k1_MSFT_s104
                                 + 1.04854 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.669582 w_k1_ADM_s104
                                 + 0.85642 w_k1_AXP_s104
                                 + 0.520954 w_k1_DIS_s104
                                 + 0.77295 w_k1_FDX_s104
                                 + 0.765981 w_k1_IBM_s104
                                 + 0.752958 w_k1_INTC_s104
                                 + 0.636254 w_k1_KO_s104
                                 + 0.912396 w_k1_MAT_s104
                                 + 0.697481 w_k1_MSFT_s104
                                 + 0.799354 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.908104 w_k1_ADM_s104
                                 + 0.930703 w_k1_AXP_s104
                                 + 0.767826 w_k1_DIS_s104
                                 + 0.805046 w_k1_FDX_s104
                                 + 1.22306 w_k1_IBM_s104
                                 + 0.705292 w_k1_INTC_s104
                                 + 0.86926 w_k1_KO_s104 + 1.19721 w_k1_MAT_s104
                                 + 0.804211 w_k1_MSFT_s104
                                 + 1.01641 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.973259 w_k1_ADM_s104
                                 + 1.08562 w_k1_AXP_s104
                                 + 0.921529 w_k1_DIS_s104
                                 + 0.996358 w_k1_FDX_s104
                                 + 1.07095 w_k1_IBM_s104
                                 + 0.944482 w_k1_INTC_s104
                                 + 0.973799 w_k1_KO_s104
                                 + 1.04797 w_k1_MAT_s104
                                 + 0.986321 w_k1_MSFT_s104
                                 + 1.13787 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.691543 w_k1_ADM_s104
                                 + 0.721097 w_k1_AXP_s104
                                 + 0.99366 w_k1_DIS_s104
                                 + 0.836305 w_k1_FDX_s104
                                 + 1.32767 w_k1_IBM_s104
                                 + 0.936731 w_k1_INTC_s104
                                 + 1.08877 w_k1_KO_s104 + 1.03362 w_k1_MAT_s104
                                 + 0.977572 w_k1_MSFT_s104
                                 + 1.02619 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.874632 w_k1_ADM_s104
                                 + 0.791613 w_k1_AXP_s104
                                 + 0.796521 w_k1_DIS_s104
                                 + 0.819633 w_k1_FDX_s104
                                 + 0.790986 w_k1_IBM_s104
                                 + 0.89753 w_k1_INTC_s104
                                 + 0.843765 w_k1_KO_s104
                                 + 0.794297 w_k1_MAT_s104
                                 + 0.82729 w_k1_MSFT_s104
                                 + 0.932959 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.93823 w_k1_ADM_s104
                                 + 0.968336 w_k1_AXP_s104
                                 + 0.90207 w_k1_DIS_s104
                                 + 1.04346 w_k1_FDX_s104
                                 + 0.70634 w_k1_IBM_s104
                                 + 0.775938 w_k1_INTC_s104
                                 + 0.777043 w_k1_KO_s104
                                 + 0.791613 w_k1_MAT_s104
                                 + 0.904512 w_k1_MSFT_s104
                                 + 0.901534 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08796 w_k1_ADM_s105
                                 + 1.07603 w_k1_AXP_s105
                                 + 1.02776 w_k1_DIS_s105
                                 + 1.13967 w_k1_FDX_s105
                                 + 0.81517 w_k1_IBM_s105
                                 + 0.910396 w_k1_INTC_s105
                                 + 0.990445 w_k1_KO_s105 + 1.3382 w_k1_MAT_s105
                                 + 0.949273 w_k1_MSFT_s105
                                 + 0.865096 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.859415 w_k1_ADM_s105
                                 + 0.82523 w_k1_AXP_s105
                                 + 1.17113 w_k1_DIS_s105
                                 + 0.934501 w_k1_FDX_s105
                                 + 1.06727 w_k1_IBM_s105
                                 + 1.06932 w_k1_INTC_s105
                                 + 1.14928 w_k1_KO_s105 + 1.57153 w_k1_MAT_s105
                                 + 1.17414 w_k1_MSFT_s105
                                 + 1.02748 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.963704 w_k1_ADM_s105
                                 + 0.768858 w_k1_AXP_s105
                                 + 0.842646 w_k1_DIS_s105
                                 + 0.906242 w_k1_FDX_s105
                                 + 0.691688 w_k1_IBM_s105
                                 + 0.744969 w_k1_INTC_s105
                                 + 0.832525 w_k1_KO_s105
                                 + 0.769287 w_k1_MAT_s105
                                 + 0.874725 w_k1_MSFT_s105
                                 + 0.850818 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.80865 w_k1_ADM_s105
                                 + 0.693167 w_k1_AXP_s105
                                 + 0.626543 w_k1_DIS_s105
                                 + 0.77355 w_k1_FDX_s105
                                 + 0.862995 w_k1_IBM_s105
                                 + 0.731951 w_k1_INTC_s105
                                 + 0.839778 w_k1_KO_s105
                                 + 1.05688 w_k1_MAT_s105
                                 + 0.799878 w_k1_MSFT_s105
                                 + 0.87041 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.986941 w_k1_ADM_s105
                                 + 0.910328 w_k1_AXP_s105
                                 + 0.871363 w_k1_DIS_s105
                                 + 0.91793 w_k1_FDX_s105
                                 + 1.14146 w_k1_IBM_s105
                                 + 0.886432 w_k1_INTC_s105
                                 + 0.986055 w_k1_KO_s105
                                 + 1.15786 w_k1_MAT_s105
                                 + 0.839543 w_k1_MSFT_s105
                                 + 0.878104 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.23509 w_k1_ADM_s105
                                 + 1.29395 w_k1_AXP_s105
                                 + 1.35799 w_k1_DIS_s105
                                 + 1.30477 w_k1_FDX_s105
                                 + 1.03736 w_k1_IBM_s105
                                 + 0.973941 w_k1_INTC_s105
                                 + 1.11248 w_k1_KO_s105 + 2.08091 w_k1_MAT_s105
                                 + 1.13276 w_k1_MSFT_s105
                                 + 1.09656 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.798633 w_k1_ADM_s105
                                 + 0.619904 w_k1_AXP_s105
                                 + 0.850124 w_k1_DIS_s105
                                 + 0.861662 w_k1_FDX_s105
                                 + 0.958592 w_k1_IBM_s105
                                 + 0.999149 w_k1_INTC_s105
                                 + 0.942698 w_k1_KO_s105
                                 + 0.845356 w_k1_MAT_s105
                                 + 0.993133 w_k1_MSFT_s105
                                 + 0.836005 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.39465 w_k1_ADM_s105
                                 + 1.3873 w_k1_AXP_s105 + 1.38505 w_k1_DIS_s105
                                 + 1.2601 w_k1_FDX_s105 + 1.56511 w_k1_IBM_s105
                                 + 1.165 w_k1_INTC_s105 + 1.48853 w_k1_KO_s105
                                 + 1.66542 w_k1_MAT_s105
                                 + 1.20732 w_k1_MSFT_s105
                                 + 1.23708 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.04449 w_k1_ADM_s106
                                 + 0.991616 w_k1_AXP_s106
                                 + 1.15942 w_k1_DIS_s106
                                 + 0.983856 w_k1_FDX_s106
                                 + 1.20215 w_k1_IBM_s106
                                 + 0.898794 w_k1_INTC_s106
                                 + 1.15159 w_k1_KO_s106 + 1.19868 w_k1_MAT_s106
                                 + 1.01392 w_k1_MSFT_s106
                                 + 1.07588 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06137 w_k1_ADM_s106
                                 + 1.14558 w_k1_AXP_s106
                                 + 1.29166 w_k1_DIS_s106
                                 + 1.29699 w_k1_FDX_s106
                                 + 0.830225 w_k1_IBM_s106
                                 + 1.14926 w_k1_INTC_s106
                                 + 0.929614 w_k1_KO_s106
                                 + 1.09129 w_k1_MAT_s106
                                 + 1.13228 w_k1_MSFT_s106
                                 + 0.96517 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.872653 w_k1_ADM_s106
                                 + 0.716111 w_k1_AXP_s106
                                 + 0.620008 w_k1_DIS_s106
                                 + 0.697334 w_k1_FDX_s106
                                 + 0.88322 w_k1_IBM_s106
                                 + 0.658957 w_k1_INTC_s106
                                 + 0.725776 w_k1_KO_s106
                                 + 1.05276 w_k1_MAT_s106
                                 + 0.684924 w_k1_MSFT_s106
                                 + 0.772255 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.953111 w_k1_ADM_s106
                                 + 0.770558 w_k1_AXP_s106
                                 + 0.771888 w_k1_DIS_s106
                                 + 0.891563 w_k1_FDX_s106
                                 + 0.631374 w_k1_IBM_s106
                                 + 0.803476 w_k1_INTC_s106
                                 + 0.778631 w_k1_KO_s106
                                 + 0.805418 w_k1_MAT_s106
                                 + 0.893521 w_k1_MSFT_s106
                                 + 0.868412 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.35393 w_k1_ADM_s106
                                 + 1.53387 w_k1_AXP_s106
                                 + 1.29646 w_k1_DIS_s106
                                 + 1.07756 w_k1_FDX_s106
                                 + 1.52888 w_k1_IBM_s106
                                 + 1.26288 w_k1_INTC_s106
                                 + 1.29869 w_k1_KO_s106 + 1.90654 w_k1_MAT_s106
                                 + 1.13461 w_k1_MSFT_s106
                                 + 1.08813 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.39551 w_k1_ADM_s106
                                 + 1.66343 w_k1_AXP_s106 + 1.9597 w_k1_DIS_s106
                                 + 1.38795 w_k1_FDX_s106
                                 + 1.74133 w_k1_IBM_s106
                                 + 1.21205 w_k1_INTC_s106
                                 + 1.39172 w_k1_KO_s106 + 2.18776 w_k1_MAT_s106
                                 + 1.28867 w_k1_MSFT_s106
                                 + 1.16827 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.41164 w_k1_ADM_s106
                                 + 1.48543 w_k1_AXP_s106
                                 + 1.42085 w_k1_DIS_s106
                                 + 1.29931 w_k1_FDX_s106
                                 + 1.09203 w_k1_IBM_s106
                                 + 1.17365 w_k1_INTC_s106
                                 + 1.13538 w_k1_KO_s106 + 1.51442 w_k1_MAT_s106
                                 + 1.04835 w_k1_MSFT_s106
                                 + 1.05507 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.02969 w_k1_ADM_s106
                                 + 0.873088 w_k1_AXP_s106
                                 + 1.06283 w_k1_DIS_s106
                                 + 0.939506 w_k1_FDX_s106
                                 + 0.987179 w_k1_IBM_s106
                                 + 0.798802 w_k1_INTC_s106
                                 + 0.939456 w_k1_KO_s106
                                 + 1.19399 w_k1_MAT_s106
                                 + 0.877713 w_k1_MSFT_s106
                                 + 0.948923 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.1514 w_k1_ADM_s107
                                 + 0.953108 w_k1_AXP_s107
                                 + 1.37976 w_k1_DIS_s107
                                 + 1.15705 w_k1_FDX_s107 + 1.0093 w_k1_IBM_s107
                                 + 1.22877 w_k1_INTC_s107
                                 + 1.15604 w_k1_KO_s107
                                 + 0.857876 w_k1_MAT_s107
                                 + 1.28355 w_k1_MSFT_s107
                                 + 1.04383 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.21035 w_k1_ADM_s107
                                 + 1.2421 w_k1_AXP_s107
                                 + 0.767049 w_k1_DIS_s107 + 1.014 w_k1_FDX_s107
                                 + 0.890494 w_k1_IBM_s107
                                 + 0.990507 w_k1_INTC_s107
                                 + 0.874102 w_k1_KO_s107
                                 + 1.11748 w_k1_MAT_s107
                                 + 0.972826 w_k1_MSFT_s107
                                 + 1.04164 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.888812 w_k1_ADM_s107
                                 + 0.657143 w_k1_AXP_s107
                                 + 0.68772 w_k1_DIS_s107
                                 + 0.78012 w_k1_FDX_s107
                                 + 0.790716 w_k1_IBM_s107
                                 + 0.733079 w_k1_INTC_s107
                                 + 0.754346 w_k1_KO_s107
                                 + 0.589434 w_k1_MAT_s107
                                 + 0.814789 w_k1_MSFT_s107
                                 + 0.846914 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.36007 w_k1_ADM_s107
                                 + 1.01332 w_k1_AXP_s107
                                 + 0.990775 w_k1_DIS_s107
                                 + 1.08439 w_k1_FDX_s107
                                 + 0.957971 w_k1_IBM_s107
                                 + 0.97048 w_k1_INTC_s107
                                 + 1.00146 w_k1_KO_s107
                                 + 0.838543 w_k1_MAT_s107
                                 + 1.15662 w_k1_MSFT_s107
                                 + 1.13347 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.4803 w_k1_ADM_s107
                                 + 1.39067 w_k1_AXP_s107
                                 + 1.45498 w_k1_DIS_s107
                                 + 1.28292 w_k1_FDX_s107
                                 + 1.27131 w_k1_IBM_s107
                                 + 1.26654 w_k1_INTC_s107
                                 + 1.39726 w_k1_KO_s107 + 1.22626 w_k1_MAT_s107
                                 + 1.28754 w_k1_MSFT_s107
                                 + 1.09427 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.913625 w_k1_ADM_s107
                                 + 0.795894 w_k1_AXP_s107
                                 + 0.952771 w_k1_DIS_s107
                                 + 1.0023 w_k1_FDX_s107
                                 + 0.762757 w_k1_IBM_s107
                                 + 0.940261 w_k1_INTC_s107
                                 + 0.890745 w_k1_KO_s107
                                 + 0.585758 w_k1_MAT_s107
                                 + 0.999532 w_k1_MSFT_s107
                                 + 0.952409 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.03922 w_k1_ADM_s107
                                 + 1.23051 w_k1_AXP_s107
                                 + 1.42342 w_k1_DIS_s107
                                 + 1.15182 w_k1_FDX_s107
                                 + 1.65952 w_k1_IBM_s107
                                 + 1.15518 w_k1_INTC_s107
                                 + 1.37989 w_k1_KO_s107 + 1.1131 w_k1_MAT_s107
                                 + 1.14326 w_k1_MSFT_s107
                                 + 1.07461 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.11778 w_k1_ADM_s107
                                 + 1.05468 w_k1_AXP_s107
                                 + 0.948068 w_k1_DIS_s107
                                 + 0.886867 w_k1_FDX_s107
                                 + 1.28462 w_k1_IBM_s107
                                 + 0.98549 w_k1_INTC_s107
                                 + 1.03359 w_k1_KO_s107
                                 + 0.866187 w_k1_MAT_s107
                                 + 1.06371 w_k1_MSFT_s107
                                 + 0.975728 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20096 w_k1_ADM_s110
                                 + 1.14184 w_k1_AXP_s110
                                 + 1.05414 w_k1_DIS_s110
                                 + 1.12211 w_k1_FDX_s110
                                 + 0.918831 w_k1_IBM_s110
                                 + 0.985744 w_k1_INTC_s110
                                 + 1.0115 w_k1_KO_s110 + 1.23366 w_k1_MAT_s110
                                 + 1.2308 w_k1_MSFT_s110 + 1.07933 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.990026 w_k1_ADM_s110
                                 + 1.22088 w_k1_AXP_s110
                                 + 1.03831 w_k1_DIS_s110
                                 + 0.956264 w_k1_FDX_s110
                                 + 1.43084 w_k1_IBM_s110
                                 + 0.831201 w_k1_INTC_s110
                                 + 1.06518 w_k1_KO_s110 + 1.42913 w_k1_MAT_s110
                                 + 0.91317 w_k1_MSFT_s110
                                 + 0.948205 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.00403 w_k1_ADM_s110
                                 + 0.959655 w_k1_AXP_s110
                                 + 0.772511 w_k1_DIS_s110
                                 + 0.923932 w_k1_FDX_s110
                                 + 0.792324 w_k1_IBM_s110
                                 + 0.875999 w_k1_INTC_s110
                                 + 0.833696 w_k1_KO_s110
                                 + 1.12813 w_k1_MAT_s110
                                 + 1.01859 w_k1_MSFT_s110
                                 + 0.950466 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.17667 w_k1_ADM_s110
                                 + 1.30306 w_k1_AXP_s110
                                 + 1.44078 w_k1_DIS_s110
                                 + 1.35653 w_k1_FDX_s110
                                 + 1.06259 w_k1_IBM_s110
                                 + 1.09201 w_k1_INTC_s110
                                 + 1.11469 w_k1_KO_s110 + 1.33005 w_k1_MAT_s110
                                 + 1.2816 w_k1_MSFT_s110 + 1.08145 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.87566 w_k1_ADM_s110
                                 + 0.919391 w_k1_AXP_s110
                                 + 0.66705 w_k1_DIS_s110
                                 + 0.734749 w_k1_FDX_s110
                                 + 0.977878 w_k1_IBM_s110
                                 + 0.721171 w_k1_INTC_s110
                                 + 0.873309 w_k1_KO_s110
                                 + 0.74191 w_k1_MAT_s110
                                 + 0.792651 w_k1_MSFT_s110
                                 + 0.895014 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.889062 w_k1_ADM_s110
                                 + 0.962796 w_k1_AXP_s110
                                 + 0.937761 w_k1_DIS_s110
                                 + 0.973619 w_k1_FDX_s110
                                 + 1.07074 w_k1_IBM_s110
                                 + 0.977706 w_k1_INTC_s110
                                 + 1.00405 w_k1_KO_s110
                                 + 0.817966 w_k1_MAT_s110
                                 + 0.964009 w_k1_MSFT_s110
                                 + 1.00038 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.649188 w_k1_ADM_s110
                                 + 0.665818 w_k1_AXP_s110
                                 + 0.615396 w_k1_DIS_s110
                                 + 0.715428 w_k1_FDX_s110
                                 + 0.74028 w_k1_IBM_s110
                                 + 0.778071 w_k1_INTC_s110
                                 + 0.762256 w_k1_KO_s110
                                 + 0.690943 w_k1_MAT_s110
                                 + 0.828274 w_k1_MSFT_s110
                                 + 0.810499 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.04333 w_k1_ADM_s110
                                 + 0.977201 w_k1_AXP_s110
                                 + 0.996438 w_k1_DIS_s110
                                 + 1.07498 w_k1_FDX_s110
                                 + 0.63559 w_k1_IBM_s110
                                 + 1.03179 w_k1_INTC_s110
                                 + 0.757115 w_k1_KO_s110
                                 + 0.521201 w_k1_MAT_s110
                                 + 1.03324 w_k1_MSFT_s110
                                 + 0.931928 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.09867 w_k1_ADM_s111
                                 + 0.996834 w_k1_AXP_s111
                                 + 1.48605 w_k1_DIS_s111
                                 + 1.32835 w_k1_FDX_s111
                                 + 1.06278 w_k1_IBM_s111
                                 + 1.1317 w_k1_INTC_s111 + 1.19919 w_k1_KO_s111
                                 + 1.20488 w_k1_MAT_s111
                                 + 1.17438 w_k1_MSFT_s111
                                 + 1.15281 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.18181 w_k1_ADM_s111
                                 + 1.05123 w_k1_AXP_s111
                                 + 0.984361 w_k1_DIS_s111
                                 + 1.04753 w_k1_FDX_s111 + 1.0704 w_k1_IBM_s111
                                 + 0.830053 w_k1_INTC_s111 + 1.149 w_k1_KO_s111
                                 + 0.92539 w_k1_MAT_s111
                                 + 0.966414 w_k1_MSFT_s111
                                 + 1.0641 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.860926 w_k1_ADM_s111
                                 + 0.580611 w_k1_AXP_s111
                                 + 0.516125 w_k1_DIS_s111
                                 + 0.658604 w_k1_FDX_s111
                                 + 0.785636 w_k1_IBM_s111
                                 + 0.714567 w_k1_INTC_s111
                                 + 0.837628 w_k1_KO_s111
                                 + 0.686161 w_k1_MAT_s111
                                 + 0.740459 w_k1_MSFT_s111
                                 + 0.779053 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.20072 w_k1_ADM_s111
                                 + 1.10779 w_k1_AXP_s111
                                 + 1.26052 w_k1_DIS_s111
                                 + 1.09007 w_k1_FDX_s111
                                 + 1.58473 w_k1_IBM_s111
                                 + 1.04397 w_k1_INTC_s111
                                 + 1.22849 w_k1_KO_s111 + 1.12076 w_k1_MAT_s111
                                 + 1.02132 w_k1_MSFT_s111
                                 + 1.18513 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.841543 w_k1_ADM_s111
                                 + 0.875612 w_k1_AXP_s111
                                 + 0.807296 w_k1_DIS_s111
                                 + 0.825013 w_k1_FDX_s111
                                 + 1.14881 w_k1_IBM_s111
                                 + 0.74744 w_k1_INTC_s111
                                 + 1.02723 w_k1_KO_s111 + 1.30465 w_k1_MAT_s111
                                 + 0.784249 w_k1_MSFT_s111
                                 + 0.97429 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.13086 w_k1_ADM_s111
                                 + 0.877456 w_k1_AXP_s111
                                 + 0.920043 w_k1_DIS_s111
                                 + 1.01258 w_k1_FDX_s111
                                 + 0.759732 w_k1_IBM_s111
                                 + 0.850634 w_k1_INTC_s111
                                 + 0.854847 w_k1_KO_s111
                                 + 0.854583 w_k1_MAT_s111
                                 + 0.847092 w_k1_MSFT_s111
                                 + 0.965892 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.872518 w_k1_ADM_s111
                                 + 0.882372 w_k1_AXP_s111
                                 + 0.710607 w_k1_DIS_s111
                                 + 0.795536 w_k1_FDX_s111
                                 + 1.31549 w_k1_IBM_s111
                                 + 0.960999 w_k1_INTC_s111
                                 + 0.901575 w_k1_KO_s111
                                 + 1.16437 w_k1_MAT_s111
                                 + 0.796106 w_k1_MSFT_s111
                                 + 0.813186 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.873978 w_k1_ADM_s111
                                 + 0.861441 w_k1_AXP_s111
                                 + 0.83139 w_k1_DIS_s111
                                 + 0.928756 w_k1_FDX_s111
                                 + 0.93667 w_k1_IBM_s111
                                 + 0.904193 w_k1_INTC_s111
                                 + 0.937363 w_k1_KO_s111
                                 + 1.13703 w_k1_MAT_s111
                                 + 0.76789 w_k1_MSFT_s111
                                 + 0.84218 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.09097 w_k1_ADM_s112
                                 + 1.0308 w_k1_AXP_s112 + 1.058 w_k1_DIS_s112
                                 + 1.08883 w_k1_FDX_s112
                                 + 0.873636 w_k1_IBM_s112
                                 + 1.08734 w_k1_INTC_s112
                                 + 1.16555 w_k1_KO_s112
                                 + 0.772275 w_k1_MAT_s112
                                 + 1.08238 w_k1_MSFT_s112
                                 + 0.939517 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.712218 w_k1_ADM_s112
                                 + 1.10005 w_k1_AXP_s112 + 1.1226 w_k1_DIS_s112
                                 + 0.929059 w_k1_FDX_s112
                                 + 1.56934 w_k1_IBM_s112
                                 + 0.939127 w_k1_INTC_s112
                                 + 1.5395 w_k1_KO_s112 + 1.49333 w_k1_MAT_s112
                                 + 1.03736 w_k1_MSFT_s112
                                 + 1.01615 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.848386 w_k1_ADM_s112
                                 + 0.846783 w_k1_AXP_s112
                                 + 0.566584 w_k1_DIS_s112
                                 + 0.781434 w_k1_FDX_s112
                                 + 0.713488 w_k1_IBM_s112
                                 + 0.938698 w_k1_INTC_s112
                                 + 0.79983 w_k1_KO_s112
                                 + 0.591441 w_k1_MAT_s112
                                 + 0.828691 w_k1_MSFT_s112
                                 + 0.709546 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.876575 w_k1_ADM_s112
                                 + 0.771575 w_k1_AXP_s112
                                 + 0.789159 w_k1_DIS_s112
                                 + 0.764249 w_k1_FDX_s112
                                 + 1.00571 w_k1_IBM_s112
                                 + 0.771192 w_k1_INTC_s112
                                 + 1.04932 w_k1_KO_s112
                                 + 0.932398 w_k1_MAT_s112
                                 + 0.777084 w_k1_MSFT_s112
                                 + 0.850003 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.02349 w_k1_ADM_s112
                                 + 1.10519 w_k1_AXP_s112
                                 + 0.810523 w_k1_DIS_s112
                                 + 0.944695 w_k1_FDX_s112
                                 + 0.676973 w_k1_IBM_s112
                                 + 1.02113 w_k1_INTC_s112
                                 + 0.884338 w_k1_KO_s112
                                 + 0.72584 w_k1_MAT_s112
                                 + 0.951926 w_k1_MSFT_s112
                                 + 0.835224 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.01734 w_k1_ADM_s112
                                 + 0.891616 w_k1_AXP_s112
                                 + 1.04119 w_k1_DIS_s112
                                 + 1.10749 w_k1_FDX_s112
                                 + 0.559546 w_k1_IBM_s112
                                 + 0.969962 w_k1_INTC_s112
                                 + 0.980138 w_k1_KO_s112
                                 + 0.731306 w_k1_MAT_s112
                                 + 1.05972 w_k1_MSFT_s112
                                 + 0.94247 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.05911 w_k1_ADM_s112
                                 + 1.10676 w_k1_AXP_s112
                                 + 1.54163 w_k1_DIS_s112
                                 + 1.31812 w_k1_FDX_s112
                                 + 1.06059 w_k1_IBM_s112
                                 + 1.08315 w_k1_INTC_s112
                                 + 1.36638 w_k1_KO_s112 + 1.14124 w_k1_MAT_s112
                                 + 1.30176 w_k1_MSFT_s112
                                 + 1.08768 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.804063 w_k1_ADM_s112
                                 + 1.20484 w_k1_AXP_s112
                                 + 1.04798 w_k1_DIS_s112
                                 + 0.99004 w_k1_FDX_s112
                                 + 1.11943 w_k1_IBM_s112
                                 + 1.14103 w_k1_INTC_s112
                                 + 1.15067 w_k1_KO_s112 + 1.36151 w_k1_MAT_s112
                                 + 0.969971 w_k1_MSFT_s112
                                 + 0.933475 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.999113 w_k1_ADM_s113
                                 + 1.12275 w_k1_AXP_s113
                                 + 1.06211 w_k1_DIS_s113
                                 + 0.965596 w_k1_FDX_s113
                                 + 1.39743 w_k1_IBM_s113
                                 + 1.02456 w_k1_INTC_s113
                                 + 1.07538 w_k1_KO_s113 + 1.46422 w_k1_MAT_s113
                                 + 0.951216 w_k1_MSFT_s113
                                 + 1.03178 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.811009 w_k1_ADM_s113
                                 + 0.821695 w_k1_AXP_s113
                                 + 0.916629 w_k1_DIS_s113
                                 + 0.997231 w_k1_FDX_s113
                                 + 0.703018 w_k1_IBM_s113
                                 + 0.995893 w_k1_INTC_s113
                                 + 0.815842 w_k1_KO_s113
                                 + 0.561641 w_k1_MAT_s113
                                 + 1.02614 w_k1_MSFT_s113
                                 + 0.863419 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.25968 w_k1_ADM_s113
                                 + 1.09764 w_k1_AXP_s113
                                 + 1.64314 w_k1_DIS_s113
                                 + 1.44672 w_k1_FDX_s113
                                 + 1.16211 w_k1_IBM_s113
                                 + 1.27198 w_k1_INTC_s113
                                 + 1.22024 w_k1_KO_s113 + 1.15257 w_k1_MAT_s113
                                 + 1.39777 w_k1_MSFT_s113
                                 + 1.17556 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.07321 w_k1_ADM_s113
                                 + 1.3009 w_k1_AXP_s113 + 1.38708 w_k1_DIS_s113
                                 + 1.15585 w_k1_FDX_s113
                                 + 1.71838 w_k1_IBM_s113
                                 + 0.923335 w_k1_INTC_s113
                                 + 1.16244 w_k1_KO_s113 + 1.68419 w_k1_MAT_s113
                                 + 0.990283 w_k1_MSFT_s113
                                 + 1.15075 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.11604 w_k1_ADM_s113
                                 + 1.01635 w_k1_AXP_s113
                                 + 1.14539 w_k1_DIS_s113
                                 + 1.22954 w_k1_FDX_s113
                                 + 0.900034 w_k1_IBM_s113
                                 + 1.13083 w_k1_INTC_s113
                                 + 0.965514 w_k1_KO_s113
                                 + 0.972653 w_k1_MAT_s113
                                 + 1.15695 w_k1_MSFT_s113
                                 + 1.01711 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.02699 w_k1_ADM_s113
                                 + 0.919328 w_k1_AXP_s113
                                 + 0.841203 w_k1_DIS_s113
                                 + 0.969687 w_k1_FDX_s113
                                 + 1.11465 w_k1_IBM_s113
                                 + 0.943236 w_k1_INTC_s113
                                 + 0.83914 w_k1_KO_s113 + 1.17773 w_k1_MAT_s113
                                 + 1.03876 w_k1_MSFT_s113
                                 + 0.88978 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.6561 w_k1_ADM_s113
                                 + 0.751355 w_k1_AXP_s113
                                 + 0.653429 w_k1_DIS_s113
                                 + 0.839749 w_k1_FDX_s113
                                 + 0.501945 w_k1_IBM_s113
                                 + 0.785776 w_k1_INTC_s113
                                 + 0.580681 w_k1_KO_s113
                                 + 0.589489 w_k1_MAT_s113
                                 + 0.86263 w_k1_MSFT_s113
                                 + 0.759244 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.95542 w_k1_ADM_s113
                                 + 0.714124 w_k1_AXP_s113
                                 + 0.644458 w_k1_DIS_s113
                                 + 0.757715 w_k1_FDX_s113
                                 + 0.835564 w_k1_IBM_s113
                                 + 0.66947 w_k1_INTC_s113
                                 + 0.769824 w_k1_KO_s113
                                 + 0.789516 w_k1_MAT_s113
                                 + 0.876037 w_k1_MSFT_s113
                                 + 0.817906 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.956763 w_k1_ADM_s114
                                 + 0.910256 w_k1_AXP_s114
                                 + 0.952208 w_k1_DIS_s114
                                 + 0.792558 w_k1_FDX_s114
                                 + 1.13043 w_k1_IBM_s114
                                 + 0.746031 w_k1_INTC_s114
                                 + 1.00577 w_k1_KO_s114 + 0.91042 w_k1_MAT_s114
                                 + 0.832825 w_k1_MSFT_s114
                                 + 0.879214 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.998127 w_k1_ADM_s114
                                 + 1.1415 w_k1_AXP_s114 + 1.38838 w_k1_DIS_s114
                                 + 1.033 w_k1_FDX_s114 + 1.46071 w_k1_IBM_s114
                                 + 0.92952 w_k1_INTC_s114 + 1.265 w_k1_KO_s114
                                 + 1.57466 w_k1_MAT_s114
                                 + 1.01958 w_k1_MSFT_s114
                                 + 1.01044 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.35044 w_k1_ADM_s114
                                 + 1.12238 w_k1_AXP_s114
                                 + 1.30185 w_k1_DIS_s114
                                 + 1.32241 w_k1_FDX_s114
                                 + 0.848458 w_k1_IBM_s114
                                 + 1.23208 w_k1_INTC_s114
                                 + 1.01666 w_k1_KO_s114
                                 + 0.909844 w_k1_MAT_s114
                                 + 1.24152 w_k1_MSFT_s114
                                 + 1.06964 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.18442 w_k1_ADM_s114
                                 + 1.0074 w_k1_AXP_s114 + 1.09739 w_k1_DIS_s114
                                 + 1.14573 w_k1_FDX_s114
                                 + 0.690339 w_k1_IBM_s114
                                 + 0.881423 w_k1_INTC_s114
                                 + 0.87195 w_k1_KO_s114
                                 + 0.927778 w_k1_MAT_s114
                                 + 1.13617 w_k1_MSFT_s114 + 1.0066 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.16471 w_k1_ADM_s114
                                 + 0.983649 w_k1_AXP_s114
                                 + 0.98031 w_k1_DIS_s114
                                 + 0.955688 w_k1_FDX_s114
                                 + 1.08553 w_k1_IBM_s114
                                 + 0.948283 w_k1_INTC_s114
                                 + 1.06386 w_k1_KO_s114 + 1.01158 w_k1_MAT_s114
                                 + 1.05758 w_k1_MSFT_s114
                                 + 0.927609 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.940906 w_k1_ADM_s114
                                 + 0.667061 w_k1_AXP_s114
                                 + 0.637024 w_k1_DIS_s114
                                 + 0.716517 w_k1_FDX_s114
                                 + 0.945881 w_k1_IBM_s114
                                 + 0.866945 w_k1_INTC_s114
                                 + 0.758635 w_k1_KO_s114
                                 + 0.644032 w_k1_MAT_s114
                                 + 0.774354 w_k1_MSFT_s114
                                 + 0.772673 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.796995 w_k1_ADM_s114
                                 + 0.769179 w_k1_AXP_s114
                                 + 0.768905 w_k1_DIS_s114
                                 + 0.909494 w_k1_FDX_s114
                                 + 0.729807 w_k1_IBM_s114
                                 + 0.94612 w_k1_INTC_s114
                                 + 0.798026 w_k1_KO_s114
                                 + 0.695641 w_k1_MAT_s114
                                 + 0.917546 w_k1_MSFT_s114
                                 + 0.867038 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.40949 w_k1_ADM_s114
                                 + 1.11006 w_k1_AXP_s114
                                 + 1.86683 w_k1_DIS_s114
                                 + 1.37699 w_k1_FDX_s114
                                 + 1.11554 w_k1_IBM_s114
                                 + 1.06338 w_k1_INTC_s114
                                 + 1.26823 w_k1_KO_s114 + 1.39866 w_k1_MAT_s114
                                 + 1.34403 w_k1_MSFT_s114
                                 + 1.07619 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.672185 w_k1_ADM_s115
                                 + 0.654398 w_k1_AXP_s115
                                 + 0.573276 w_k1_DIS_s115
                                 + 0.747634 w_k1_FDX_s115
                                 + 0.68792 w_k1_IBM_s115
                                 + 0.85232 w_k1_INTC_s115
                                 + 0.752849 w_k1_KO_s115
                                 + 0.422388 w_k1_MAT_s115
                                 + 0.742791 w_k1_MSFT_s115
                                 + 0.786561 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.11909 w_k1_ADM_s115
                                 + 1.02882 w_k1_AXP_s115
                                 + 1.08758 w_k1_DIS_s115
                                 + 0.951149 w_k1_FDX_s115
                                 + 1.43829 w_k1_IBM_s115
                                 + 0.936957 w_k1_INTC_s115
                                 + 1.16721 w_k1_KO_s115 + 1.40811 w_k1_MAT_s115
                                 + 0.914231 w_k1_MSFT_s115
                                 + 1.09474 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.913475 w_k1_ADM_s115
                                 + 0.946028 w_k1_AXP_s115
                                 + 0.833226 w_k1_DIS_s115
                                 + 0.93117 w_k1_FDX_s115
                                 + 0.799371 w_k1_IBM_s115
                                 + 0.91312 w_k1_INTC_s115
                                 + 0.810399 w_k1_KO_s115
                                 + 0.879491 w_k1_MAT_s115
                                 + 0.927607 w_k1_MSFT_s115
                                 + 0.86659 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.28321 w_k1_ADM_s115
                                 + 1.43434 w_k1_AXP_s115
                                 + 1.36657 w_k1_DIS_s115
                                 + 1.22736 w_k1_FDX_s115 + 1.1836 w_k1_IBM_s115
                                 + 0.973413 w_k1_INTC_s115
                                 + 1.25178 w_k1_KO_s115 + 1.5347 w_k1_MAT_s115
                                 + 1.09092 w_k1_MSFT_s115
                                 + 1.14862 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.13217 w_k1_ADM_s115
                                 + 1.15855 w_k1_AXP_s115 + 2.0479 w_k1_DIS_s115
                                 + 1.37903 w_k1_FDX_s115 + 1.1652 w_k1_IBM_s115
                                 + 1.15027 w_k1_INTC_s115
                                 + 1.58678 w_k1_KO_s115 + 1.31283 w_k1_MAT_s115
                                 + 1.37749 w_k1_MSFT_s115
                                 + 1.21814 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.1047 w_k1_ADM_s115
                                 + 0.857613 w_k1_AXP_s115
                                 + 1.0469 w_k1_DIS_s115 + 1.11616 w_k1_FDX_s115
                                 + 0.863721 w_k1_IBM_s115
                                 + 0.993628 w_k1_INTC_s115
                                 + 0.894742 w_k1_KO_s115
                                 + 0.795595 w_k1_MAT_s115
                                 + 1.04024 w_k1_MSFT_s115
                                 + 0.963505 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.09434 w_k1_ADM_s115
                                 + 0.96101 w_k1_AXP_s115
                                 + 1.42411 w_k1_DIS_s115
                                 + 1.32109 w_k1_FDX_s115
                                 + 0.852345 w_k1_IBM_s115
                                 + 1.1555 w_k1_INTC_s115 + 1.03365 w_k1_KO_s115
                                 + 0.660779 w_k1_MAT_s115
                                 + 1.19663 w_k1_MSFT_s115
                                 + 1.10061 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.905174 w_k1_ADM_s115
                                 + 0.771831 w_k1_AXP_s115
                                 + 1.03421 w_k1_DIS_s115
                                 + 0.833343 w_k1_FDX_s115
                                 + 1.02264 w_k1_IBM_s115
                                 + 0.866397 w_k1_INTC_s115
                                 + 1.05705 w_k1_KO_s115
                                 + 0.827458 w_k1_MAT_s115
                                 + 0.933464 w_k1_MSFT_s115
                                 + 0.906409 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.990918 w_k1_ADM_s116
                                 + 1.18769 w_k1_AXP_s116
                                 + 1.31216 w_k1_DIS_s116
                                 + 1.23256 w_k1_FDX_s116
                                 + 0.991139 w_k1_IBM_s116
                                 + 1.13211 w_k1_INTC_s116
                                 + 0.937275 w_k1_KO_s116
                                 + 0.671955 w_k1_MAT_s116
                                 + 1.14745 w_k1_MSFT_s116
                                 + 1.12336 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.749884 w_k1_ADM_s116
                                 + 0.660644 w_k1_AXP_s116
                                 + 0.586819 w_k1_DIS_s116
                                 + 0.718592 w_k1_FDX_s116
                                 + 0.7613 w_k1_IBM_s116
                                 + 0.798625 w_k1_INTC_s116
                                 + 0.556844 w_k1_KO_s116
                                 + 0.388916 w_k1_MAT_s116
                                 + 0.64743 w_k1_MSFT_s116
                                 + 0.776242 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.20272 w_k1_ADM_s116
                                 + 1.09466 w_k1_AXP_s116
                                 + 0.97264 w_k1_DIS_s116
                                 + 1.26052 w_k1_FDX_s116
                                 + 0.666809 w_k1_IBM_s116
                                 + 1.14291 w_k1_INTC_s116
                                 + 0.726518 w_k1_KO_s116
                                 + 0.585561 w_k1_MAT_s116
                                 + 1.04091 w_k1_MSFT_s116
                                 + 0.995907 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.790657 w_k1_ADM_s116
                                 + 1.04196 w_k1_AXP_s116
                                 + 0.950301 w_k1_DIS_s116
                                 + 0.94296 w_k1_FDX_s116
                                 + 1.16688 w_k1_IBM_s116
                                 + 0.979655 w_k1_INTC_s116
                                 + 0.860485 w_k1_KO_s116
                                 + 1.17049 w_k1_MAT_s116
                                 + 0.918545 w_k1_MSFT_s116
                                 + 1.02406 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.899317 w_k1_ADM_s116
                                 + 0.71291 w_k1_AXP_s116
                                 + 0.673335 w_k1_DIS_s116
                                 + 0.802026 w_k1_FDX_s116
                                 + 0.907796 w_k1_IBM_s116
                                 + 0.897359 w_k1_INTC_s116
                                 + 0.762455 w_k1_KO_s116
                                 + 0.565025 w_k1_MAT_s116
                                 + 0.773787 w_k1_MSFT_s116
                                 + 0.816332 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.48237 w_k1_ADM_s116
                                 + 1.22101 w_k1_AXP_s116
                                 + 1.79986 w_k1_DIS_s116
                                 + 1.58932 w_k1_FDX_s116
                                 + 0.978471 w_k1_IBM_s116
                                 + 1.20547 w_k1_INTC_s116
                                 + 1.21483 w_k1_KO_s116
                                 + 0.803631 w_k1_MAT_s116
                                 + 1.40899 w_k1_MSFT_s116
                                 + 1.34632 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.980262 w_k1_ADM_s116
                                 + 0.903241 w_k1_AXP_s116
                                 + 0.850119 w_k1_DIS_s116
                                 + 1.03014 w_k1_FDX_s116
                                 + 0.757825 w_k1_IBM_s116
                                 + 1.08349 w_k1_INTC_s116
                                 + 0.753752 w_k1_KO_s116
                                 + 0.508881 w_k1_MAT_s116
                                 + 0.936011 w_k1_MSFT_s116
                                 + 0.872784 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.929179 w_k1_ADM_s116
                                 + 1.36175 w_k1_AXP_s116
                                 + 1.30645 w_k1_DIS_s116
                                 + 1.12567 w_k1_FDX_s116 + 1.536 w_k1_IBM_s116
                                 + 0.956315 w_k1_INTC_s116
                                 + 1.10779 w_k1_KO_s116 + 1.45009 w_k1_MAT_s116
                                 + 1.11364 w_k1_MSFT_s116
                                 + 1.25149 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.23736 w_k1_ADM_s117
                                 + 1.13556 w_k1_AXP_s117
                                 + 0.815507 w_k1_DIS_s117
                                 + 0.932015 w_k1_FDX_s117
                                 + 0.786881 w_k1_IBM_s117
                                 + 1.0871 w_k1_INTC_s117
                                 + 0.980745 w_k1_KO_s117
                                 + 0.720879 w_k1_MAT_s117
                                 + 1.02727 w_k1_MSFT_s117
                                 + 0.972294 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.30486 w_k1_ADM_s117
                                 + 1.32129 w_k1_AXP_s117
                                 + 1.60804 w_k1_DIS_s117
                                 + 1.36581 w_k1_FDX_s117
                                 + 0.911482 w_k1_IBM_s117
                                 + 1.36269 w_k1_INTC_s117
                                 + 1.30274 w_k1_KO_s117
                                 + 0.922817 w_k1_MAT_s117
                                 + 1.60426 w_k1_MSFT_s117 + 1.2051 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.945163 w_k1_ADM_s117
                                 + 0.667593 w_k1_AXP_s117
                                 + 0.552394 w_k1_DIS_s117
                                 + 0.723016 w_k1_FDX_s117
                                 + 0.783772 w_k1_IBM_s117
                                 + 0.890389 w_k1_INTC_s117
                                 + 0.77048 w_k1_KO_s117
                                 + 0.697032 w_k1_MAT_s117
                                 + 0.87448 w_k1_MSFT_s117
                                 + 0.763345 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03756 w_k1_ADM_s117
                                 + 0.831538 w_k1_AXP_s117
                                 + 0.80315 w_k1_DIS_s117
                                 + 0.908931 w_k1_FDX_s117
                                 + 1.04673 w_k1_IBM_s117
                                 + 1.04383 w_k1_INTC_s117
                                 + 0.942637 w_k1_KO_s117
                                 + 0.946892 w_k1_MAT_s117
                                 + 0.945793 w_k1_MSFT_s117
                                 + 1.0404 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0064 w_k1_ADM_s117
                                 + 1.21582 w_k1_AXP_s117
                                 + 1.30334 w_k1_DIS_s117
                                 + 1.18289 w_k1_FDX_s117
                                 + 1.23486 w_k1_IBM_s117
                                 + 1.21633 w_k1_INTC_s117
                                 + 1.40593 w_k1_KO_s117 + 1.46794 w_k1_MAT_s117
                                 + 1.32246 w_k1_MSFT_s117
                                 + 1.05907 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.23667 w_k1_ADM_s117
                                 + 1.026 w_k1_AXP_s117 + 0.976189 w_k1_DIS_s117
                                 + 0.924501 w_k1_FDX_s117
                                 + 1.57549 w_k1_IBM_s117
                                 + 1.00337 w_k1_INTC_s117
                                 + 1.34176 w_k1_KO_s117 + 1.14409 w_k1_MAT_s117
                                 + 0.997238 w_k1_MSFT_s117
                                 + 1.10311 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.9926 w_k1_ADM_s117
                                 + 0.829149 w_k1_AXP_s117
                                 + 0.88434 w_k1_DIS_s117
                                 + 1.00556 w_k1_FDX_s117
                                 + 0.846662 w_k1_IBM_s117
                                 + 1.10656 w_k1_INTC_s117
                                 + 1.03068 w_k1_KO_s117
                                 + 0.714417 w_k1_MAT_s117
                                 + 1.19845 w_k1_MSFT_s117
                                 + 1.00666 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.00998 w_k1_ADM_s117
                                 + 0.716491 w_k1_AXP_s117
                                 + 0.789327 w_k1_DIS_s117
                                 + 0.90558 w_k1_FDX_s117
                                 + 0.623788 w_k1_IBM_s117
                                 + 0.830327 w_k1_INTC_s117
                                 + 0.768775 w_k1_KO_s117
                                 + 0.452535 w_k1_MAT_s117
                                 + 1.00338 w_k1_MSFT_s117
                                 + 0.860571 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0147507664258163 w_k1_ADM_s000
                                 + 0.0233156283959672 w_k1_ADM_s001
                                 + 0.0179398256083577 w_k1_ADM_s002
                                 + 0.027119024018043 w_k1_ADM_s003
                                 + 0.0515422501342939 w_k1_ADM_s004
                                 + 0.0187108748674914 w_k1_ADM_s005
                                 + 0.04731884860512 w_k1_ADM_s006
                                 + 0.0339056769247433 w_k1_ADM_s007
                                 + 0.0181425461835101 w_k1_ADM_s010
                                 + 0.0361528182968777 w_k1_ADM_s011
                                 + 0.0117921800845776 w_k1_ADM_s012
                                 + 0.0593652337097044 w_k1_ADM_s013
                                 + 0.0282448771425734 w_k1_ADM_s014
                                 + 0.0195722584802827 w_k1_ADM_s015
                                 + 0.0335015974442084 w_k1_ADM_s016
                                 + 0.0172682126554388 w_k1_ADM_s017
                                 + 0.0565368832417322 w_k1_ADM_s100
                                 + 0.0232530436740021 w_k1_ADM_s101
                                 + 0.0432480184593859 w_k1_ADM_s102
                                 + 0.0391011516580495 w_k1_ADM_s103
                                 + 0.0391119680227838 w_k1_ADM_s104
                                 + 0.0301168619078702 w_k1_ADM_s105
                                 + 0.0431791217908297 w_k1_ADM_s106
                                 + 0.0582506916147455 w_k1_ADM_s107
                                 + 0.0374506204139514 w_k1_ADM_s110
                                 + 0.0306390101482186 w_k1_ADM_s111
                                 + 0.0127866393116744 w_k1_ADM_s112
                                 + 0.0292386680037959 w_k1_ADM_s113
                                 + 0.0266856163233402 w_k1_ADM_s114
                                 + 0.0246154490939752 w_k1_ADM_s115
                                 + 0.0502510742204999 w_k1_ADM_s116
                                 + 0.0315393204845779 w_k1_ADM_s117
                                 + 0.013710963114695 w_k1_AXP_s000
                                 + 0.0242830829607913 w_k1_AXP_s001
                                 + 0.0147888056647909 w_k1_AXP_s002
                                 + 0.0233199053188431 w_k1_AXP_s003
                                 + 0.0595184537255623 w_k1_AXP_s004
                                 + 0.0211936632426988 w_k1_AXP_s005
                                 + 0.0393417717131653 w_k1_AXP_s006
                                 + 0.0328749256428917 w_k1_AXP_s007
                                 + 0.0194316519378705 w_k1_AXP_s010
                                 + 0.0305592734883101 w_k1_AXP_s011
                                 + 0.0138096274793121 w_k1_AXP_s012
                                 + 0.0464839887460419 w_k1_AXP_s013
                                 + 0.026107157577685 w_k1_AXP_s014
                                 + 0.0223916219430657 w_k1_AXP_s015
                                 + 0.0354517092917625 w_k1_AXP_s016
                                 + 0.0164743156840185 w_k1_AXP_s017
                                 + 0.0600651547937046 w_k1_AXP_s100
                                 + 0.02401891685273 w_k1_AXP_s101
                                 + 0.0435433175278698 w_k1_AXP_s102
                                 + 0.0392321778406633 w_k1_AXP_s103
                                 + 0.0466268734083804 w_k1_AXP_s104
                                 + 0.0288188173464797 w_k1_AXP_s105
                                 + 0.0414759339118634 w_k1_AXP_s106
                                 + 0.051378047000507 w_k1_AXP_s107
                                 + 0.0401548012303282 w_k1_AXP_s110
                                 + 0.0275457343124713 w_k1_AXP_s111
                                 + 0.014012414291954 w_k1_AXP_s112
                                 + 0.0280123000112596 w_k1_AXP_s113
                                 + 0.0243867264140226 w_k1_AXP_s114
                                 + 0.0237230027712397 w_k1_AXP_s115
                                 + 0.0536753235084719 w_k1_AXP_s116
                                 + 0.0284581115708681 w_k1_AXP_s117
                                 + 0.0142295571507105 w_k1_DIS_s000
                                 + 0.025924024807436 w_k1_DIS_s001
                                 + 0.0161419899639209 w_k1_DIS_s002
                                 + 0.0228671127350905 w_k1_DIS_s003
                                 + 0.0524209068157299 w_k1_DIS_s004
                                 + 0.02122343204817 w_k1_DIS_s005
                                 + 0.0445991436261176 w_k1_DIS_s006
                                 + 0.0334257273651362 w_k1_DIS_s007
                                 + 0.0208720517668029 w_k1_DIS_s010
                                 + 0.0358227563028147 w_k1_DIS_s011
                                 + 0.0136168608473439 w_k1_DIS_s012
                                 + 0.0487338271206189 w_k1_DIS_s013
                                 + 0.0250646253484508 w_k1_DIS_s014
                                 + 0.0221326731260139 w_k1_DIS_s015
                                 + 0.0354067492298078 w_k1_DIS_s016
                                 + 0.0176115322949389 w_k1_DIS_s017
                                 + 0.0624698953823056 w_k1_DIS_s100
                                 + 0.0239098386624739 w_k1_DIS_s101
                                 + 0.0418229456085792 w_k1_DIS_s102
                                 + 0.0376813649626872 w_k1_DIS_s103
                                 + 0.0422703357771485 w_k1_DIS_s104
                                 + 0.0297759008734545 w_k1_DIS_s105
                                 + 0.0435519079084481 w_k1_DIS_s106
                                 + 0.0549413419416657 w_k1_DIS_s107
                                 + 0.0374683576908688 w_k1_DIS_s110
                                 + 0.030334104663366 w_k1_DIS_s111
                                 + 0.0142870220932808 w_k1_DIS_s112
                                 + 0.0291623207004252 w_k1_DIS_s113
                                 + 0.0282948965463117 w_k1_DIS_s114
                                 + 0.0260440080041404 w_k1_DIS_s115
                                 + 0.0529771939252576 w_k1_DIS_s116
                                 + 0.0287128578296245 w_k1_DIS_s117
                                 + 0.0146168745218161 w_k1_FDX_s000
                                 + 0.0255010318296377 w_k1_FDX_s001
                                 + 0.0157120443433885 w_k1_FDX_s002
                                 + 0.0227972756361902 w_k1_FDX_s003
                                 + 0.0539846481409302 w_k1_FDX_s004
                                 + 0.021381769921924 w_k1_FDX_s005
                                 + 0.0418424794734661 w_k1_FDX_s006
                                 + 0.0312847254069066 w_k1_FDX_s007
                                 + 0.019185918682314 w_k1_FDX_s010
                                 + 0.0329270418729749 w_k1_FDX_s011
                                 + 0.0132660342288879 w_k1_FDX_s012
                                 + 0.0499433975184393 w_k1_FDX_s013
                                 + 0.026157188850231 w_k1_FDX_s014
                                 + 0.0211769632135085 w_k1_FDX_s015
                                 + 0.0346692696033551 w_k1_FDX_s016
                                 + 0.0160432002541139 w_k1_FDX_s017
                                 + 0.0600416633977577 w_k1_FDX_s100
                                 + 0.0239309512960863 w_k1_FDX_s101
                                 + 0.0414970874095773 w_k1_FDX_s102
                                 + 0.0396477800410039 w_k1_FDX_s103
                                 + 0.0421210936668776 w_k1_FDX_s104
                                 + 0.0299292906705325 w_k1_FDX_s105
                                 + 0.0401037547652872 w_k1_FDX_s106
                                 + 0.0535043186168955 w_k1_FDX_s107
                                 + 0.0382116592474492 w_k1_FDX_s110
                                 + 0.0301606570819833 w_k1_FDX_s111
                                 + 0.0137437139678201 w_k1_FDX_s112
                                 + 0.0298625387904889 w_k1_FDX_s113
                                 + 0.0252581490166933 w_k1_FDX_s114
                                 + 0.0245021011958873 w_k1_FDX_s115
                                 + 0.0548233037847996 w_k1_FDX_s116
                                 + 0.0293702988266972 w_k1_FDX_s117
                                 + 0.0143334006668171 w_k1_IBM_s000
                                 + 0.025928954266173 w_k1_IBM_s001
                                 + 0.0156892894713008 w_k1_IBM_s002
                                 + 0.0230867290438412 w_k1_IBM_s003
                                 + 0.0526196317042917 w_k1_IBM_s004
                                 + 0.021627917491499 w_k1_IBM_s005
                                 + 0.0420254081316025 w_k1_IBM_s006
                                 + 0.0339209750469899 w_k1_IBM_s007
                                 + 0.0196927092771454 w_k1_IBM_s010
                                 + 0.0327108837875075 w_k1_IBM_s011
                                 + 0.0131778495687414 w_k1_IBM_s012
                                 + 0.0515932693402693 w_k1_IBM_s013
                                 + 0.0272896522014348 w_k1_IBM_s014
                                 + 0.0217815793709592 w_k1_IBM_s015
                                 + 0.0353391276171473 w_k1_IBM_s016
                                 + 0.0167499192368586 w_k1_IBM_s017
                                 + 0.061187243050328 w_k1_IBM_s100
                                 + 0.025181316948082 w_k1_IBM_s101
                                 + 0.0429337282217504 w_k1_IBM_s102
                                 + 0.0394475001558264 w_k1_IBM_s103
                                 + 0.0435492506608242 w_k1_IBM_s104
                                 + 0.0297248778463824 w_k1_IBM_s105
                                 + 0.0415344357545022 w_k1_IBM_s106
                                 + 0.0543743402608578 w_k1_IBM_s107
                                 + 0.0392303959439871 w_k1_IBM_s110
                                 + 0.0319609959364267 w_k1_IBM_s111
                                 + 0.0139046243645851 w_k1_IBM_s112
                                 + 0.0309995133700362 w_k1_IBM_s113
                                 + 0.0266021719062389 w_k1_IBM_s114
                                 + 0.0239820210167995 w_k1_IBM_s115
                                 + 0.0539280160932026 w_k1_IBM_s116
                                 + 0.030447597505204 w_k1_IBM_s117
                                 + 0.0140018692494497 w_k1_INTC_s000
                                 + 0.0231659405894778 w_k1_INTC_s001
                                 + 0.0162890985772891 w_k1_INTC_s002
                                 + 0.0241438251976537 w_k1_INTC_s003
                                 + 0.0544238960355025 w_k1_INTC_s004
                                 + 0.0233350260201417 w_k1_INTC_s005
                                 + 0.0426830882503368 w_k1_INTC_s006
                                 + 0.0387173609203097 w_k1_INTC_s007
                                 + 0.0183347147275048 w_k1_INTC_s010
                                 + 0.0295414108195496 w_k1_INTC_s011
                                 + 0.0117099653802662 w_k1_INTC_s012
                                 + 0.0489898844139261 w_k1_INTC_s013
                                 + 0.0248059675010887 w_k1_INTC_s014
                                 + 0.0218243165528163 w_k1_INTC_s015
                                 + 0.0356855966135301 w_k1_INTC_s016
                                 + 0.0174452230478499 w_k1_INTC_s017
                                 + 0.057229999705051 w_k1_INTC_s100
                                 + 0.0289969532067863 w_k1_INTC_s101
                                 + 0.0415303243845488 w_k1_INTC_s102
                                 + 0.0368217511681228 w_k1_INTC_s103
                                 + 0.0403792895730427 w_k1_INTC_s104
                                 + 0.026679025950331 w_k1_INTC_s105
                                 + 0.0369155431608544 w_k1_INTC_s106
                                 + 0.0527798979708947 w_k1_INTC_s107
                                 + 0.0351159866398909 w_k1_INTC_s110
                                 + 0.0273225146725722 w_k1_INTC_s111
                                 + 0.0137141041352168 w_k1_INTC_s112
                                 + 0.0281465909662581 w_k1_INTC_s113
                                 + 0.0235022512688822 w_k1_INTC_s114
                                 + 0.022585993733934 w_k1_INTC_s115
                                 + 0.0534170971435448 w_k1_INTC_s116
                                 + 0.0317112782131139 w_k1_INTC_s117
                                 + 0.0135612508166057 w_k1_KO_s000
                                 + 0.0280870275174677 w_k1_KO_s001
                                 + 0.0160036731717874 w_k1_KO_s002
                                 + 0.0238524002868573 w_k1_KO_s003
                                 + 0.0470126950360979 w_k1_KO_s004
                                 + 0.02442614111061 w_k1_KO_s005
                                 + 0.0399321633138106 w_k1_KO_s006
                                 + 0.0357549846287489 w_k1_KO_s007
                                 + 0.0190604405150597 w_k1_KO_s010
                                 + 0.0301144860767669 w_k1_KO_s011
                                 + 0.014347721917155 w_k1_KO_s012
                                 + 0.0491959583802391 w_k1_KO_s013
                                 + 0.0240700734186477 w_k1_KO_s014
                                 + 0.0224628958381236 w_k1_KO_s015
                                 + 0.0359512122581089 w_k1_KO_s016
                                 + 0.0168435655199702 w_k1_KO_s017
                                 + 0.0558266349862669 w_k1_KO_s100
                                 + 0.0236134540002864 w_k1_KO_s101
                                 + 0.039345900295259 w_k1_KO_s102
                                 + 0.0342461091299335 w_k1_KO_s103
                                 + 0.0412169028199418 w_k1_KO_s104
                                 + 0.0301845492473993 w_k1_KO_s105
                                 + 0.0393213216470037 w_k1_KO_s106
                                 + 0.0541471113014843 w_k1_KO_s107
                                 + 0.0369757776723504 w_k1_KO_s110
                                 + 0.0308544778999988 w_k1_KO_s111
                                 + 0.0159741098048563 w_k1_KO_s112
                                 + 0.0270185574446668 w_k1_KO_s113
                                 + 0.0258801928972942 w_k1_KO_s114
                                 + 0.0244785164180922 w_k1_KO_s115
                                 + 0.045447650211566 w_k1_KO_s116
                                 + 0.0322091612240562 w_k1_KO_s117
                                 + 0.0145190068233243 w_k1_MAT_s000
                                 + 0.0306758892330199 w_k1_MAT_s001
                                 + 0.0117494334919066 w_k1_MAT_s002
                                 + 0.0189025501466822 w_k1_MAT_s003
                                 + 0.0522693799534278 w_k1_MAT_s004
                                 + 0.0215893963875766 w_k1_MAT_s005
                                 + 0.0369191583463748 w_k1_MAT_s006
                                 + 0.0327120005162673 w_k1_MAT_s007
                                 + 0.0217274856144188 w_k1_MAT_s010
                                 + 0.0297715417116421 w_k1_MAT_s011
                                 + 0.015035644978963 w_k1_MAT_s012
                                 + 0.0452438759922006 w_k1_MAT_s013
                                 + 0.0279700511110422 w_k1_MAT_s014
                                 + 0.024782634133543 w_k1_MAT_s015
                                 + 0.0450476869365466 w_k1_MAT_s016
                                 + 0.0188981041850048 w_k1_MAT_s017
                                 + 0.0607569745123133 w_k1_MAT_s100
                                 + 0.0341985244020653 w_k1_MAT_s101
                                 + 0.0427648611237796 w_k1_MAT_s102
                                 + 0.0356398560851413 w_k1_MAT_s103
                                 + 0.0432135113770171 w_k1_MAT_s104
                                 + 0.0403337711042749 w_k1_MAT_s105
                                 + 0.0504351988586326 w_k1_MAT_s106
                                 + 0.0444463414368949 w_k1_MAT_s107
                                 + 0.039684129281762 w_k1_MAT_s110
                                 + 0.0315779275545058 w_k1_MAT_s111
                                 + 0.0141107570338 w_k1_MAT_s112
                                 + 0.0313812889179524 w_k1_MAT_s113
                                 + 0.026636660032978 w_k1_MAT_s114
                                 + 0.0245397728875448 w_k1_MAT_s115
                                 + 0.0442658325851278 w_k1_MAT_s116
                                 + 0.0285592702277849 w_k1_MAT_s117
                                 + 0.0140533906609329 w_k1_MSFT_s000
                                 + 0.02443021509689 w_k1_MSFT_s001
                                 + 0.0155638850029578 w_k1_MSFT_s002
                                 + 0.022227286310364 w_k1_MSFT_s003
                                 + 0.0485519656964259 w_k1_MSFT_s004
                                 + 0.022206364893328 w_k1_MSFT_s005
                                 + 0.0445334899568662 w_k1_MSFT_s006
                                 + 0.0347164809361655 w_k1_MSFT_s007
                                 + 0.018224978328762 w_k1_MSFT_s010
                                 + 0.0318496247032823 w_k1_MSFT_s011
                                 + 0.0123668763199458 w_k1_MSFT_s012
                                 + 0.0507273238124471 w_k1_MSFT_s013
                                 + 0.0239189140706573 w_k1_MSFT_s014
                                 + 0.0215156135563573 w_k1_MSFT_s015
                                 + 0.0380108971345904 w_k1_MSFT_s016
                                 + 0.0169489018455348 w_k1_MSFT_s017
                                 + 0.0615120578937267 w_k1_MSFT_s100
                                 + 0.0239387516046059 w_k1_MSFT_s101
                                 + 0.0387468200524292 w_k1_MSFT_s102
                                 + 0.0367818475942885 w_k1_MSFT_s103
                                 + 0.0407493814243271 w_k1_MSFT_s104
                                 + 0.0283663659917413 w_k1_MSFT_s105
                                 + 0.0380725567698357 w_k1_MSFT_s106
                                 + 0.055920274914522 w_k1_MSFT_s107
                                 + 0.0386886647748781 w_k1_MSFT_s110
                                 + 0.0273564303046871 w_k1_MSFT_s111
                                 + 0.0138922221493241 w_k1_MSFT_s112
                                 + 0.0300818578975302 w_k1_MSFT_s113
                                 + 0.0255191595522809 w_k1_MSFT_s114
                                 + 0.0235521170732341 w_k1_MSFT_s115
                                 + 0.0511768657155408 w_k1_MSFT_s116
                                 + 0.0328032506725822 w_k1_MSFT_s117
                                 + 0.0141244659981432 w_k1_PG_s000
                                 + 0.0247072527272626 w_k1_PG_s001
                                 + 0.0154790030181001 w_k1_PG_s002
                                 + 0.0217337923532209 w_k1_PG_s003
                                 + 0.0536355149527073 w_k1_PG_s004
                                 + 0.0206856935857207 w_k1_PG_s005
                                 + 0.0400393291711235 w_k1_PG_s006
                                 + 0.0330786349062583 w_k1_PG_s007
                                 + 0.0197939151076613 w_k1_PG_s010
                                 + 0.0327678952534243 w_k1_PG_s011
                                 + 0.0119526326883465 w_k1_PG_s012
                                 + 0.050372760738836 w_k1_PG_s013
                                 + 0.0249551484232998 w_k1_PG_s014
                                 + 0.0213760659794215 w_k1_PG_s015
                                 + 0.0347716469772385 w_k1_PG_s016
                                 + 0.0163361823129435 w_k1_PG_s017
                                 + 0.0598711738128767 w_k1_PG_s100
                                 + 0.0226758154341481 w_k1_PG_s101
                                 + 0.0376762264920076 w_k1_PG_s102
                                 + 0.0376833346421691 w_k1_PG_s103
                                 + 0.0439286879069187 w_k1_PG_s104
                                 + 0.0277529415011764 w_k1_PG_s105
                                 + 0.0383295063423412 w_k1_PG_s106
                                 + 0.0523867669464947 w_k1_PG_s107
                                 + 0.037249952102916 w_k1_PG_s110
                                 + 0.0297983058050518 w_k1_PG_s111
                                 + 0.0128108562316571 w_k1_PG_s112
                                 + 0.0278333812668753 w_k1_PG_s113
                                 + 0.0237415045962717 w_k1_PG_s114
                                 + 0.0235662699933281 w_k1_PG_s115
                                 + 0.0535323915536625 w_k1_PG_s116
                                 + 0.0298158693024503 w_k1_PG_s117 >= 1.15
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
