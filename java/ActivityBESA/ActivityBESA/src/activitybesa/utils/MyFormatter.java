/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package activitybesa.utils;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.Date;
import java.util.logging.LogRecord;

/**
 *
 * @author mauricio
 */

class MyFormatter extends Formatter {
    // Create a DateFormat to format the logger timestamp.
    private static final DateFormat df = new SimpleDateFormat("dd/MM/yyyy hh:mm:ss.SSS");

    @Override
    public String format(LogRecord record) {
        
        StackTraceElement[] e = Thread.currentThread().getStackTrace();
        int line = e[8].getLineNumber();

        String className = record.getSourceClassName();
        String[] classParts = className.split("\\.");
        
        
        if (classParts.length > 0) {
            className = classParts[classParts.length - 1];
        } else {
            className = record.getSourceClassName();
        }
        
        StringBuilder builder = new StringBuilder(1000);
        builder.append(df.format(new Date(record.getMillis()))).append("-");
        builder.append("[").append(className).append(".");
        builder.append(record.getSourceMethodName()).append("]" + line);
        builder.append("[").append(record.getLevel()).append("] - ");
        builder.append(formatMessage(record));
        builder.append("\n");
        
        
        
        return builder.toString();
    }

    @Override
    public String getHead(Handler h) {
        return super.getHead(h);
    }

    @Override
    public String getTail(Handler h) {
        return super.getTail(h);
    }
}