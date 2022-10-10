package com.coursera_assignments.rajas.courseramodernartuiassignment;

import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.content.DialogInterface;
import android.os.Bundle;

/**
 * Created by Rajas on 05/31/2015.
 */
public class MoreInfo extends DialogFragment
{
    @Override
    public Dialog onCreateDialog ( Bundle savedInstanceState ) {

        AlertDialog.Builder builder = new AlertDialog.Builder( getActivity() );
        builder.setMessage( R.string.dialog_text ).setPositiveButton( R.string.dialog_visit,
                new DialogInterface.OnClickListener() {

                    /**
                     * This method will be invoked when the positive button in the dialog
                     * is clicked.
                     * <p/>
                     *
                     * @param dialog The dialog that received the click.
                     * @param id     The button that was clicked (the position
                     *               of the item clicked.)
                     */
                    public void onClick ( DialogInterface dialog, int id ) {

                        ((FragmentDialog)getActivity()).doPositiveClick();
                    }
                } ).setNegativeButton( R.string.dialog_not_now,
                new DialogInterface.OnClickListener() {

                    /**
                     * This method will be invoked when the negative button in the dialog
                     * is clicked.
                     * <p/>
                     *
                     * @param dialog The dialog that received the click.
                     * @param id     The button that was clicked (the position
                     *               of the item clicked.)
                     */
                    public void onClick ( DialogInterface dialog, int id ) {

                        ((FragmentDialog)getActivity()).doNegativeClick();
                    }
                }  );

        return builder.create();
    }
}
